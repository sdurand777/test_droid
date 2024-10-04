#include <cstdio>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

// #include "utils.cuh"

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<long>> graph_t;
typedef std::vector<torch::Tensor> tensor_list_t;



#define MIN_DEPTH 0.25

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)


#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)


// fonction pour la somme accumulee du tableau sdata qui represente un warp soit 32 threads volatile pour ne pas optimiser sdata car plusieurs threads accedent aux data simultanement
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}


// fonction pour sommer toutes les valeurs de sdata
__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  // on arrive au niveau du warp on en profite
  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}



// applique une rotation a partir du quaternion q au vecteur X pour obtenir Y
__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}


// applique la rotation SO3 puis la translation t a X pour obtenir Y
__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  // normalement X[3] vaut 1 scale factor pour ajouter t a Y
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}



// on applique adjoint a X pour obtenir Y attention X a 6 coords position et velocity en fait X et Y sont des jacobiens
// on connait le jaocbien X Jj et on va determiner Y Ji dans le code 
// adjoin tde SE3 matrice 6x6 avec R t^R puis 0 R
// ici on utilise la formulation avec les quaternions pour appliquer les rotations
__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
    // inversion du quaternion rotation inverse
    float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
    // application de la rotation inverse a X position
    actSO3(qinv, &X[0], &Y[0]);
    // application de la rotation inverse a X velocity
    actSO3(qinv, &X[3], &Y[3]);

    // compute u
    float u[3], v[3];
    u[0] = t[2]*X[1] - t[1]*X[2];
    u[1] = t[0]*X[2] - t[2]*X[0];
    u[2] = t[1]*X[0] - t[0]*X[1];

    // compute v
    actSO3(qinv, u, v);

    // compute final Y
    Y[3] += v[0];
    Y[4] += v[1];
    Y[5] += v[2];
}




// compute transfo entre deux frame S33
__device__ void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {

    // relative quaternion
    qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
    qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
    qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
    qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

    // relative translation
    actSO3(qij, ti, tij);
    tij[0] = tj[0] - tij[0];
    tij[1] = tj[1] - tij[1];
    tij[2] = tj[2] - tij[2];
}

  
// compute quaternion from phi
__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  // gestion cas phi proche de 0
  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  // compute quaternion
  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;
}



// produit vectoriel a b et update b with the result
__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}




// exponential map for SE3 conver xi en t and q xi linear algebra element to linear group SE3 elements q t on remplace R par q
__device__ void
expSE3(const float *xi, float* t, float* q) {
    // SE3 exponential map
    // get q from phi inside xi the last 3 coords of xi
    expSO3(xi + 3, q);
    // xi translation part
    float tau[3] = {xi[0], xi[1], xi[2]};
    // xi rotation part
    float phi[3] = {xi[3], xi[4], xi[5]};

    float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
    float theta = sqrtf(theta_sq);

    // compute t from tau
    t[0] = tau[0]; 
    t[1] = tau[1]; 
    t[2] = tau[2];

    // case phi  very small
    if (theta > 1e-4) {
        float a = (1 - cosf(theta)) / theta_sq;
        crossInplace(phi, tau);
        t[0] += a * tau[0];
        t[1] += a * tau[1];
        t[2] += a * tau[2];

        float b = (theta - sinf(theta)) / (theta * theta_sq);
        crossInplace(phi, tau);
        t[0] += b * tau[0];
        t[1] += b * tau[1];
        t[2] += b * tau[2];
    }
}




// build Hessian rhs vector 
__global__ void projective_transform_kernel(
        const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
        const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
        const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
        torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
    // recuperation indices block et thread
    // on parallelize
    // chaque block_id correspond a un edge du graph
    const int block_id = blockIdx.x;
    // les threads sont utilise au sein du block pour faire les operations en paralleles
    const int thread_id = threadIdx.x;

    // size disparite
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    // indice dans le graph
    int ix = static_cast<int>(ii[block_id]);
    int jx = static_cast<int>(jj[block_id]);

    // shared data
    __shared__ float fx;
    __shared__ float fy;
    __shared__ float cx;
    __shared__ float cy;

    // relative pose for stereo frames
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];

    // load intrinsics from global memory
    if (thread_id == 0) {
        fx = intrinsics[0];
        fy = intrinsics[1];
        cx = intrinsics[2];
        cy = intrinsics[3];
    }

    __syncthreads();

    // stereo frames same id
    if (ix == jx) {
        // stereo cas we fix the relative transformation between left and right
        if (thread_id == 0) {
            tij[0] =  -0.1;
            tij[1] =     0;
            tij[2] =     0;
            qij[0] =     0;
            qij[1] =     0;
            qij[2] =     0;
            qij[3] =     1;
        }
    }

    else {

        // load poses from global memory for two different lefts
        // poses 7 dimensional vector translation quaternion

        // collect translations for frame i and j
        if (thread_id < 3) {
            ti[thread_id] = poses[ix][thread_id];
            tj[thread_id] = poses[jx][thread_id];
        }

        // collect quaternion for frame i and j
        if (thread_id < 4) {
            qi[thread_id] = poses[ix][thread_id+3];
            qj[thread_id] = poses[jx][thread_id+3];
        }

        __syncthreads();
        
        // compute relatite transfo between frame i and j
        if (thread_id == 0) {
            relSE3(ti, qi, tj, qj, tij, qij);
        }
    }

    __syncthreads();

    //points  from i and j
    float Xi[4];
    float Xj[4];

    // jacobians
    // Jx tangent 12 for frame i and j tangent element in SE3 are 6 dimensional vectors rho theta

    // En fait ici on Dti Dtj Dd soit les trois jacobiens pour ce edge i j 
    // jacobien pose
    float Jx[12];
    // jacobien depth ou disparite
    float Jz;

    // frame i tangent jacobian
    float* Ji = &Jx[0];
    // frame j tangent jacobian
    float* Jj = &Jx[6];

    // hessians hij symetrique donc uniquement la moitie
    // hessiens de 12 donc uniquement pour les poses
    float hij[12*(12+1)/2];

    // vector to build the hessian problem the right handside vector - J r where here vi and vj are the pose part for frame i and j
    float vi[6], vj[6];

    int l;
    // fill with 0
    for (l=0; l<12*(12+1)/2; l++) {
        hij[l] = 0;
    }

    // fill with 0
    for (int n=0; n<6; n++) {
        vi[n] = 0;
        vj[n] = 0;
    }

    __syncthreads();

    // we loop over all pixels ht*wd
    GPU_1D_KERNEL_LOOP(k, ht*wd) {

        // pixel within ht wd image
        const int i = k / wd;
        const int j = k % wd;

        const float u = static_cast<float>(j);
        const float v = static_cast<float>(i);

        // frame i tha t we project onto frame j 

        // homogenous coordinates for each pixel
        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1;
        Xi[3] = disps[ix][i][j];

        // transform homogenous point Xi to Xj using tij and qij soit les poses initi guess qui vont etre opitimises par le BA
        actSE3(tij, qij, Xi, Xj);

        // we het the 3D data from Xj
        const float x = Xj[0];
        const float y = Xj[1];
        const float h = Xj[3];

        // get depth from Xj and check value convert to inverse depth
        const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];

        // squered value for latter
        const float d2 = d * d;

        // confidence update based on reprojected Xj weight are from update operator convgry to weight the revised flow predicted by the convgru opertar target
        float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
        float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];

        // reprojection distance between target predicted (or revised with delta to update coords) by the convgru and reprojection based on estimated disp and pose
        const float ru = target[block_id][0][i][j] - (fx * d * x + cx);
        const float rv = target[block_id][1][i][j] - (fy * d * y + cy);

        // x - coordinate

        // compute jacobian
        // h ici est un facteur echelle qui doir valoir 1
        Jj[0] = fx * (h*d);
        Jj[1] = fx * 0;
        Jj[2] = fx * (-x*h*d2);
        Jj[3] = fx * (-x*y*d2);
        Jj[4] = fx * (1 + x*x*d2);
        Jj[5] = fx * (-y*d);

        // jaocobiien par rapport a d de projection
        Jz = fx * (tij[0] * d - tij[2] * (x * d2));

        // hessian only for depth or disparity
        Cii[block_id][k] = wu * Jz * Jz;
        // bias for depth for th right hand side vector of the optimization problem
        bz[block_id][k] = wu * ru * Jz;

        if (ix == jx) wu = 0;

        // since we have Jj we compute Ji with the adjoint
        adjSE3(tij, qij, Jj, Ji);
        for (int n=0; n<6; n++) Ji[n] *= -1;

        // update hessian part for pose with Jx
        l=0;
        for (int n=0; n<12; n++) {
            for (int m=0; m<=n; m++) {
                hij[l] += wu * Jx[n] * Jx[m];
                l++;
            }
        }

        // blocks cross pose depth
        for (int n=0; n<6; n++) {
            // vector for poses right hand side
            vi[n] += wu * ru * Ji[n];
            vj[n] += wu * ru * Jj[n];

            // block of hessian cross pose depth
            Eii[block_id][n][k] = wu * Jz * Ji[n];
            Eij[block_id][n][k] = wu * Jz * Jj[n];
        }

        // y coordinate
        
        // compute jacobian from two matrices
        // pixel derivated wrt to point
        // point derivated wrt to pose
        // to get pixel derivated wrt to pose
        Jj[0] = fy * 0;
        Jj[1] = fy * (h*d);
        Jj[2] = fy * (-y*h*d2);
        Jj[3] = fy * (-1 - y*y*d2);
        Jj[4] = fy * (x*y*d2);
        Jj[5] = fy * (x*d);
        
        // jocobien suivant depth
        Jz = fy * (tij[1] * d - tij[2] * (y * d2));

        // depth hessian block
        Cii[block_id][k] += wv * Jz * Jz;
        // bias for right hand side vector
        bz[block_id][k] += wv * rv * Jz;

        if (ix == jx) wv = 0;

        // compute Ji
        adjSE3(tij, qij, Jj, Ji);
        for (int n=0; n<6; n++) Ji[n] *= -1;

        // update hessian
        l=0;
        for (int n=0; n<12; n++) {
            for (int m=0; m<=n; m++) {
                hij[l] += wv * Jx[n] * Jx[m];
                l++;
            }
        }

        // blocks cross pose and depth
        for (int n=0; n<6; n++) {
            // right hand side vector
            vi[n] += wv * rv * Ji[n];
            vj[n] += wv * rv * Jj[n];

            // cross block for the hessian
            Eii[block_id][n][k] += wv * Jz * Ji[n];
            Eij[block_id][n][k] += wv * Jz * Jj[n];
        }


    }


    // fill vs and Hs the global Hessian and global rhs vector for the optimization problem

    __syncthreads();

    __shared__ float sdata[THREADS];
    for (int n=0; n<6; n++) {
        sdata[threadIdx.x] = vi[n];
        blockReduce(sdata);
        if (threadIdx.x == 0) {
            vs[0][block_id][n] = sdata[0];
        }

        __syncthreads();

        sdata[threadIdx.x] = vj[n];
        blockReduce(sdata);
        if (threadIdx.x == 0) {
            vs[1][block_id][n] = sdata[0];
        }

    }

    l=0;
    for (int n=0; n<12; n++) {
        for (int m=0; m<=n; m++) {
            sdata[threadIdx.x] = hij[l];
            blockReduce(sdata);

            if (threadIdx.x == 0) {
                if (n<6 && m<6) {
                    // for symetry
                    Hs[0][block_id][n][m] = sdata[0];
                    Hs[0][block_id][m][n] = sdata[0];
                }
                else if (n >=6 && m<6) {
                    Hs[1][block_id][m][n-6] = sdata[0];
                    Hs[2][block_id][n-6][m] = sdata[0];
                }
                else {
                    Hs[3][block_id][n-6][m-6] = sdata[0];
                    Hs[3][block_id][m-6][n-6] = sdata[0];
                }
            }

            l++;
        }
    }
}








// method pour recuperer la distance entre les frames
__global__ void frame_distance_kernel(
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
        const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
        torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist,
        const float beta) 
{

    // block id
    const int block_id = blockIdx.x;
    // thread id
    const int thread_id = threadIdx.x;

    // size image
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    //  id edge in the graph
    __shared__ int ix;
    __shared__ int jx;

    // intrinsics
    __shared__ float fx;
    __shared__ float fy;
    __shared__ float cx;
    __shared__ float cy;

    // transformation
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];

    // load intrinsics from global memory
    if (thread_id == 0) {
        // recuperation edge
        ix = static_cast<int>(ii[block_id]);
        jx = static_cast<int>(jj[block_id]);
        // recuperation intrinsics
        fx = intrinsics[0];
        fy = intrinsics[1];
        cx = intrinsics[2];
        cy = intrinsics[3];
    }

    __syncthreads();


    //points 
    float Xi[4];
    float Xj[4];

    // temp variables
    __shared__ float accum[THREADS]; accum[thread_id] = 0;
    __shared__ float valid[THREADS]; valid[thread_id] = 0;
    __shared__ float total[THREADS]; total[thread_id] = 0;

    __syncthreads();

    // fill data
    for (int n=0; n<1; n++) {

        // recuperation des poses
        // translation
        if (thread_id < 3) {
            ti[thread_id] = poses[ix][thread_id];
            tj[thread_id] = poses[jx][thread_id];
        }

        // quaternion
        if (thread_id < 4) {
            qi[thread_id] = poses[ix][thread_id+3];
            qj[thread_id] = poses[jx][thread_id+3];
        }

        __syncthreads();

        // relative transfo between frame i and j
        relSE3(ti, qi, tj, qj, tij, qij);

        // optical flow
        float d, du, dv;

        // loop over all pixels ht * wd
        GPU_1D_KERNEL_LOOP(k, ht*wd) {
            // recuperation pixel coords
            const int i = k / wd;
            const int j = k % wd;

            // conversion to float
            const float u = static_cast<float>(j);
            const float v = static_cast<float>(i);


            // if (disps[ix][i][j] < 0.01) {
            //   continue;
            // }

            // homogenous coordinates projection 3D du pixel avec disp
            Xi[0] = (u - cx) / fx;
            Xi[1] = (v - cy) / fy;
            Xi[2] = 1;
            Xi[3] = disps[ix][i][j];

            // transform homogenous point transfo complete rotation et translation
            // reproject Xi onto frame j to obtain Xj using relative pose tij qij
            actSE3(tij, qij, Xi, Xj);

            // optical flow
            // displacement along x
            du = fx * (Xj[0] / Xj[2]) + cx - u;
            // displacement along y
            dv = fy * (Xj[1] / Xj[2]) + cy - v;
            // disp norm
            d = sqrtf(du*du + dv*dv);

            // poids entre translation et rotation
            // partie rotation ajoute
            total[threadIdx.x] += beta;

            // on a une contrainte sur la depth de la reprojection sur la frame j
            if (Xj[2] > MIN_DEPTH) {
                accum[threadIdx.x] += beta * d;
                valid[threadIdx.x] += beta;
            }

            // recuperation a nouveau de X i useless ?
            Xi[0] = (u - cx) / fx;
            Xi[1] = (v - cy) / fy;
            Xi[2] = 1;
            Xi[3] = disps[ix][i][j];

            // translation component only
            Xj[0] = Xi[0] + Xi[3] * tij[0];
            Xj[1] = Xi[1] + Xi[3] * tij[1];
            Xj[2] = Xi[2] + Xi[3] * tij[2];

            du = fx * (Xj[0] / Xj[2]) + cx - u;
            dv = fy * (Xj[1] / Xj[2]) + cy - v;
            d = sqrtf(du*du + dv*dv);

            // partie translation ajoute
            total[threadIdx.x] += (1 - beta);

            if (Xj[2] > MIN_DEPTH) {
                accum[threadIdx.x] += (1 - beta) * d;
                valid[threadIdx.x] += (1 - beta);
            }
        }

        if (threadIdx.x == 0) {
            int tmp = ix;
            ix = jx;
            jx = tmp;
        }

        __syncthreads();

    }

    // addition des distances pour chaque pixel
    __syncthreads(); blockReduce(accum);
    __syncthreads(); blockReduce(total);
    __syncthreads(); blockReduce(valid);

    __syncthreads();

    if (thread_id == 0) {
        // on check si il y a assez de pixel valid dont depth est ok sinon on fie la distance a 1000 si ok on divise accum par valid pour obtenir la moyenne
        dist[block_id] = (valid[0] / (total[0] + 1e-8) < 0.75) ? 1000.0 : accum[0] / valid[0];
    }
}







// depth filter kernel pour le visualisateur
__global__ void depth_filter_kernel(
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
        const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds,
        const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> thresh,
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> counter)
{

    // recuperation des ids
    const int block_id = blockIdx.x;
    const int neigh_id = blockIdx.y;
    // loop over index
    const int index = blockIdx.z * blockDim.x + threadIdx.x;

    // if (threadIdx.x == 0) {
    //   printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, blockDim.x, threadIdx.x);
    // }

    // recuperation des dimensions de disps
    const int num = disps.size(0);
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    // indices du edge
    __shared__ int ix;
    __shared__ int jx;

    // intrinsics
    __shared__ float fx;
    __shared__ float fy;
    __shared__ float cx;
    __shared__ float cy;

    // poses data for the edge frame i and frame j
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];

    // intrinsics
    if (threadIdx.x == 0) {
        //  recuperation des indices
        ix = static_cast<int>(inds[block_id]);
        // on recupere jx autour de ix
        jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id;
        // recuperation des intrinsics
        fx = intrinsics[0];
        fy = intrinsics[1];
        cx = intrinsics[2];
        cy = intrinsics[3];
    }

    __syncthreads();

    if (jx < 0 || jx >= num) 
    {
        return;
    }

    const float t = thresh[block_id];

    // load poses from global memory
    // translation
    if (threadIdx.x < 3) {
        ti[threadIdx.x] = poses[ix][threadIdx.x];
        tj[threadIdx.x] = poses[jx][threadIdx.x];
    }

    // quaternions
    if (threadIdx.x < 4) {
        qi[threadIdx.x] = poses[ix][threadIdx.x+3];
        qj[threadIdx.x] = poses[jx][threadIdx.x+3];
    }

    __syncthreads();

    if (threadIdx.x == 0) 
    {
        // transfo relative
        relSE3(ti, qi, tj, qj, tij, qij);
    }

    //points 
    float Xi[4];
    float Xj[4];

    __syncthreads();

    // loop over all pixels
    if (index < ht*wd) 
    {
        // pixel coords
        const int i = index / wd;
        const int j = index % wd;

        // pixel coords convert to float
        const float ui = static_cast<float>(j);
        const float vi = static_cast<float>(i);
        // disparity info
        const float di = disps[ix][i][j];

        // homogenous coordinates projection to 3D point from pixel
        Xi[0] = (ui - cx) / fx;
        Xi[1] = (vi - cy) / fy;
        Xi[2] = 1;
        Xi[3] = di;

        // transform homogenous point apply 3D relative transfo to project Xi onto frame j to get Xj
        actSE3(tij, qij, Xi, Xj);

        // projection of 3D Xj onto frame j to get 2d coords onto frame j
        const float uj = fx * (Xj[0] / Xj[2]) + cx;
        const float vj = fy * (Xj[1] / Xj[2]) + cy;
        const float dj = Xj[3] / Xj[2];

        // pixel cpprds on frame j
        const int u0 = static_cast<int>(floor(uj));
        const int v0 = static_cast<int>(floor(vj));

        // check if pixel coords within the image shape
        if (u0 >= 0 && v0 >= 0 && u0 < wd-1 && v0 < ht-1) 
        {
            // get decimal part
            const float wx = ceil(uj) - uj;
            const float wy = ceil(vj) - vj;

            // get depth values around pixel
            const float d00 = disps[jx][v0+0][u0+0];
            const float d01 = disps[jx][v0+0][u0+1];
            const float d10 = disps[jx][v0+1][u0+0];
            const float d11 = disps[jx][v0+1][u0+1];

            // bilinear interpolation tp get depth value
            const float dj_hat = wy*wx*d00 + wy*(1-wx)*d01 + (1-wy)*wx*d10 + (1-wy)*(1-wx)*d11;

            // filter depth based on t threshold and relatively to surrounding depth values
            const float err = abs(1.0/dj - 1.0/dj_hat);
            if       (abs(1.0/dj - 1.0/d00) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
            else if  (abs(1.0/dj - 1.0/d01) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
            else if  (abs(1.0/dj - 1.0/d10) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
            else if  (abs(1.0/dj - 1.0/d11) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
        }
    }
}



//  inverse projection kernel convert disps to points 3D
__global__ void iproj_kernel(
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
        const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
        torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points)

{

    // get indices
    const int block_id = blockIdx.x;
    const int index = blockIdx.y * blockDim.x + threadIdx.x;

    // get disps shape info
    const int num = disps.size(0);
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    // intrinsics
    __shared__ float fx;
    __shared__ float fy;
    __shared__ float cx;
    __shared__ float cy;

    // pose info
    __shared__ float t[3];
    __shared__ float q[4];

    // collect intrinsics info
    if (threadIdx.x == 0) {
        fx = intrinsics[0];
        fy = intrinsics[1];
        cx = intrinsics[2];
        cy = intrinsics[3];
    }

    __syncthreads();


    // load poses from global memory
    // fill translations
    if (threadIdx.x < 3) {
        t[threadIdx.x] = poses[block_id][threadIdx.x];
    }

    // fill quaternions
    if (threadIdx.x < 4) {
        q[threadIdx.x] = poses[block_id][threadIdx.x+3];
    }

    __syncthreads();

    //points 
    float Xi[4];
    float Xj[4];

    // loop over pixels
    if (index < ht*wd) 
    {
        // get pixel coord
        const int i = index / wd;
        const int j = index % wd;

        // convert pixel coords to float
        const float ui = static_cast<float>(j);
        const float vi = static_cast<float>(i);
        // depth value
        const float di = disps[block_id][i][j];

        // homogenous coordinates project 2D pixel of frame i into 3D point Xi
        Xi[0] = (ui - cx) / fx;
        Xi[1] = (vi - cy) / fy;
        Xi[2] = 1;
        Xi[3] = di;

        // transform homogenous point
        // apply relative transfo to Xi to project Xi onto frame j in 3D
        actSE3(t, q, Xi, Xj);

        // collect Xj
        points[block_id][i][j][0] = Xj[0] / Xj[3];
        points[block_id][i][j][1] = Xj[1] / Xj[3];
        points[block_id][i][j][2] = Xj[2] / Xj[3];

    }
}




// apply update xi to update t and q to t1 and q1
__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
    // retraction on SE3 manifold

    float dt[3] = {0, 0, 0};
    float dq[4] = {0, 0, 0, 1};

    // convert xi to SE3 manifold dt dq
    expSE3(xi, dt, dq);

    // update q
    q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
    q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
    q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
    q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

    // update t
    actSO3(dq, t, t1);
    t1[0] += dt[0];
    t1[1] += dt[1];
    t1[2] += dt[2];
}



// recuperation de la poses updated avec dx using retrSE3
__global__ void pose_retr_kernel(
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
        const int t0, const int t1) 
{

    for (int k=t0+threadIdx.x; k<t1; k+=blockDim.x) {
        float xi[6], q[4], q1[4], t[3], t1[3];

        t[0] = poses[k][0];
        t[1] = poses[k][1];
        t[2] = poses[k][2];

        q[0] = poses[k][3];
        q[1] = poses[k][4];
        q[2] = poses[k][5];
        q[3] = poses[k][6];

        for (int n=0; n<6; n++) {
            xi[n] = dx[k-t0][n];
        }

        // update pose t q to t1 q1
        retrSE3(xi, t, q, t1, q1);

        poses[k][0] = t1[0];
        poses[k][1] = t1[1];
        poses[k][2] = t1[2];

        poses[k][3] = q1[0];
        poses[k][4] = q1[1];
        poses[k][5] = q1[2];
        poses[k][6] = q1[3];
    }
}




// recupreation de la disps updated avec dz
__global__ void disp_retr_kernel(
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dz,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds) 
{
    const int i = inds[blockIdx.x];
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    // loop over image pixels
    for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) 
    {
        // update d
        float d = disps[i][k/wd][k%wd] + dz[blockIdx.x][k];
        // update final disp with updated d
        disps[i][k/wd][k%wd] = d;
    }
}


// utils kernel on utilise ptrs et idxs pour mettre a jour inps on recupere outs
__global__ void accum_kernel(
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inps,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ptrs,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idxs,
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> outs)
{

    const int block_id = blockIdx.x;
    const int D = inps.size(2);

    const int start = ptrs[block_id];
    const int end = ptrs[block_id+1];

    for (int k=threadIdx.x; k<D; k+=blockDim.x) {
        float x = 0;
        for (int i=start; i<end; i++) {
            x += inps[idxs[i]][k];
        }
        outs[block_id][k] = x;
    }  
}


// utils accum_cuda on traite data avec ix et jx
torch::Tensor accum_cuda(torch::Tensor data, torch::Tensor ix, torch::Tensor jx) 
{
    // recuperation des indices
    torch::Tensor ix_cpu = ix.to(torch::kCPU);
    torch::Tensor jx_cpu = jx.to(torch::kCPU);
    // sort ix
    torch::Tensor inds = torch::argsort(ix_cpu);

    // init ptr vers les donnees
    long* ix_data = ix_cpu.data_ptr<long>();
    long* jx_data = jx_cpu.data_ptr<long>();
    long* kx_data = inds.data_ptr<long>();

    int count = jx.size(0);
    std::vector<int> cols;

    torch::Tensor ptrs_cpu = torch::zeros({count+1}, 
            torch::TensorOptions().dtype(torch::kInt64));

    long* ptrs_data = ptrs_cpu.data_ptr<long>();
    ptrs_data[0] = 0;

    int i = 0;
    for (int j=0; j<count; j++) {
        while (i < ix.size(0) && ix_data[kx_data[i]] <= jx_data[j]) {
            if (ix_data[kx_data[i]] == jx_data[j])
                cols.push_back(kx_data[i]);
            i++;
        }
        ptrs_data[j+1] = cols.size();
    }

    torch::Tensor idxs_cpu = torch::zeros({long(cols.size())}, 
            torch::TensorOptions().dtype(torch::kInt64));

    long* idxs_data = idxs_cpu.data_ptr<long>();

    for (int i=0; i<cols.size(); i++) {
        idxs_data[i] = cols[i];
    }

    torch::Tensor ptrs = ptrs_cpu.to(torch::kCUDA);
    torch::Tensor idxs = idxs_cpu.to(torch::kCUDA);

    torch::Tensor out = torch::zeros({jx.size(0), data.size(1)},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // kernel cuda pour accumuler les donnees
    accum_kernel<<<count, THREADS>>>(
            data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            ptrs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
            idxs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
            out.packed_accessor32<float,2,torch::RestrictPtrTraits>());

    return out;
}



// 
__global__ void EEt6x6_kernel(
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,

        const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> S)
{

    // indices
    const int ix = idx[blockIdx.x][0];
    const int jx = idx[blockIdx.x][1];
    const int kx = idx[blockIdx.x][2];

    // shape of E
    const int D = E.size(2);

    // matrice 6 * 6
    float dS[6][6];
    // gradient des poses ?
    float ei[6];
    float ej[6];

    // fill dS with 0
    for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
            dS[i][j] = 0;
        }
    }

    // loop over R dimension
    for (int k=threadIdx.x; k<D; k+=blockDim.x) {
        // get q data
        const float q = Q[kx][k];

        // coalesced memory read
        for (int n=0; n<6; n++) 
        {
            ei[n] = E[ix][n][k] * q;
            ej[n] = E[jx][n][k];
        }

        // block EEt
        for (int n=0; n<6; n++) {
            for (int m=0; m<6; m++) {
                dS[n][m] += ei[n] * ej[m];
            }
        }
    }

    __syncthreads();
    __shared__ float sdata[THREADS];

    // Output S
    for (int n=0; n<6; n++) {
        for (int m=0; m<6; m++) {
            sdata[threadIdx.x] = dS[n][m];

            blockReduce(sdata);

            if (threadIdx.x == 0) {
                S[blockIdx.x][n][m] = sdata[0];
            }
        }
    }
}




__global__ void Ev6x1_kernel(
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> E,
        const torch::PackedTensorAccessor32<float, 2,torch::RestrictPtrTraits> Q,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w,
        const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v)
{
    const int D = E.size(2);
    const int kx = idx[blockIdx.x][0];

    float b[6];
    for (int n=0; n<6; n++) {
        b[n] = 0.0;
    }

    for (int k=threadIdx.x; k<D; k+=blockDim.x) {
        const float q_w = Q[kx][k] * w[kx][k];

        for (int n=0; n<6; n++) {
            b[n] += q_w * E[blockIdx.x][n][k];
        }
    }

    __syncthreads();
    __shared__ float sdata[THREADS];

    for (int n=0; n<6; n++) {
        sdata[threadIdx.x] = b[n];
        blockReduce(sdata);

        if (threadIdx.x == 0) {
            v[blockIdx.x][n] += sdata[0];
        }
    }
}



__global__ void EvT6x1_kernel(
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> x,
        const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idx,
        torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w)
{

    const int D = E.size(2);
    const int ix = idx[blockIdx.x];

    if (idx[blockIdx.x] <= 0 || idx[blockIdx.x] >= x.size(0))
        return;

    for (int k=threadIdx.x; k<D; k+=blockDim.x) {
        float dw = 0;
        for (int n=0; n<6; n++) {
            dw += E[blockIdx.x][n][k] * x[ix][n];
        }
        w[blockIdx.x][k] = dw;
    }
}



// class sparseblock pour build the optimization problem
class SparseBlock {
    public:

        Eigen::SparseMatrix<double> A;
        Eigen::VectorX<double> b;

        SparseBlock(int N, int M) : N(N), M(M) {
            A = Eigen::SparseMatrix<double>(N*M, N*M);
            b = Eigen::VectorXd::Zero(N*M);
        }

        SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
                int N, int M) : A(A), b(b), N(N), M(M) {}

        // left hand side
        void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

            auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
            auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
            auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

            auto As_acc = As_cpu.accessor<double,3>();
            auto ii_acc = ii_cpu.accessor<long,1>();
            auto jj_acc = jj_cpu.accessor<long,1>();

            std::vector<T> tripletList;
            for (int n=0; n<ii.size(0); n++) {
                const int i = ii_acc[n];
                const int j = jj_acc[n];

                if (i >= 0 && j >= 0) {
                    for (int k=0; k<M; k++) {
                        for (int l=0; l<M; l++) {
                            double val = As_acc[n][k][l];
                            tripletList.push_back(T(M*i + k, M*j + l, val));
                        }
                    }
                }
            }
            A.setFromTriplets(tripletList.begin(), tripletList.end());
        }

        // right hand side
        void update_rhs(torch::Tensor bs, torch::Tensor ii) {
            auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
            auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

            auto bs_acc = bs_cpu.accessor<double,2>();
            auto ii_acc = ii_cpu.accessor<long,1>();

            for (int n=0; n<ii.size(0); n++) {
                const int i = ii_acc[n];
                if (i >= 0) {
                    for (int j=0; j<M; j++) {
                        b(i*M + j) += bs_acc[n][j];
                    }
                }
            }
        }

        SparseBlock operator-(const SparseBlock& S) {
            return SparseBlock(A - S.A, b - S.b, N, M);
        }

        std::tuple<torch::Tensor, torch::Tensor> get_dense() {
            Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

            torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
                    .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

            torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
                    .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

            return std::make_tuple(H, v);

        }

        // solve optimization problem
        torch::Tensor solve(const float lm=0.0001, const float ep=0.1) {

            torch::Tensor dx;

            Eigen::SparseMatrix<double> L(A);
            L.diagonal().array() += ep + lm * L.diagonal().array();

            Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(L);

            if (solver.info() == Eigen::Success) {
                Eigen::VectorXd x = solver.solve(b);
                dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
                        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
            }
            else {
                dx = torch::zeros({N, M}, torch::TensorOptions()
                        .device(torch::kCUDA).dtype(torch::kFloat32));
            }

            return dx;
        }

    private:
        const int N;
        const int M;

};



// schur block pour le tricks de schur
SparseBlock schur_block(torch::Tensor E,
        torch::Tensor Q,
        torch::Tensor w,
        torch::Tensor ii,
        torch::Tensor jj,
        torch::Tensor kk,
        const int t0,
        const int t1)
{

    // get indices
    torch::Tensor ii_cpu = ii.to(torch::kCPU);
    torch::Tensor jj_cpu = jj.to(torch::kCPU);
    torch::Tensor kk_cpu = kk.to(torch::kCPU);

    const int P = t1 - t0;
    const long* ii_data = ii_cpu.data_ptr<long>();
    const long* jj_data = jj_cpu.data_ptr<long>();
    const long* kk_data = kk_cpu.data_ptr<long>();

    std::vector<std::vector<long>> graph(P);
    std::vector<std::vector<long>> index(P);

    for (int n=0; n<ii_cpu.size(0); n++) {
        const int j = jj_data[n];
        const int k = kk_data[n];

        if (j >= t0 && j <= t1) {
            const int t = j - t0;
            graph[t].push_back(k);
            index[t].push_back(n);
        }
    }

    std::vector<long> ii_list, jj_list, idx, jdx;

    for (int i=0; i<P; i++) {
        for (int j=0; j<P; j++) {
            for (int k=0; k < graph[i].size(); k++) {
                for (int l=0; l < graph[j].size(); l++) {
                    if (graph[i][k] == graph[j][l]) {
                        ii_list.push_back(i);
                        jj_list.push_back(j);

                        idx.push_back(index[i][k]);
                        idx.push_back(index[j][l]);
                        idx.push_back(graph[i][k]);
                    }
                }
            }
        }
    }

    torch::Tensor ix_cuda = torch::from_blob(idx.data(), {long(idx.size())}, 
            torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA).view({-1, 3});

    torch::Tensor jx_cuda = torch::stack({kk_cpu}, -1)
        .to(torch::kCUDA).to(torch::kInt64);

    torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
            torch::TensorOptions().dtype(torch::kInt64)).view({-1});

    torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
            torch::TensorOptions().dtype(torch::kInt64)).view({-1});

    torch::Tensor S = torch::zeros({ix_cuda.size(0), 6, 6}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    torch::Tensor v = torch::zeros({jx_cuda.size(0), 6},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EEt6x6_kernel<<<ix_cuda.size(0), THREADS>>>(
            E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            ix_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
            S.packed_accessor32<float,3,torch::RestrictPtrTraits>());

    Ev6x1_kernel<<<jx_cuda.size(0), THREADS>>>(
            E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
            v.packed_accessor32<float,2,torch::RestrictPtrTraits>());

    // schur block
    SparseBlock A(P, 6);
    A.update_lhs(S, ii2_cpu, jj2_cpu);
    A.update_rhs(v, jj_cpu - t0);

    return A;
}








// main method for the BA
std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor disps_sens,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only)
{
    // // conversion to accessor
    auto targets_accessor = targets.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto weights_accessor = weights.packed_accessor32<float,4,torch::RestrictPtrTraits>();
    auto poses_accessor =  poses.packed_accessor32<float,2,torch::RestrictPtrTraits>();
    auto disps_accessor = disps.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    auto intrinsics_accessor = intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>();

    auto disps_sens_accessor = disps_sens.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    auto ii_accessor = ii.packed_accessor32<long,1,torch::RestrictPtrTraits>();
    auto jj_accessor = jj.packed_accessor32<long,1,torch::RestrictPtrTraits>();

    auto eta_accessor = eta.packed_accessor32<float,3,torch::RestrictPtrTraits>();

    auto opts = poses.options();
    const int num = ii.size(0); // nombre de edges
    const int ht = disps.size(1);// shape image
    const int wd = disps.size(2); // shape image

    torch::Tensor ts = torch::arange(t0, t1).to(torch::kCUDA); // ts fpr time frame fpr BA

    std::cout << ts << std::endl;

    torch::Tensor ii_exp = torch::cat({ts, ii}, 0); 
    torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

    std::cout << ii_exp << std::endl;
    std::cout << jj_exp << std::endl;



    std::tuple<torch::Tensor, torch::Tensor> kuniq = 
        torch::_unique(ii_exp, true, true);

    torch::Tensor kx = std::get<0>(kuniq);
    torch::Tensor kk_exp = std::get<1>(kuniq);

    torch::Tensor dx; // update pose
    torch::Tensor dz; // update depth

    // initialize buffers
    torch::Tensor Hs = torch::zeros({4, num, 6, 6}, opts); // hessain for pose
    torch::Tensor vs = torch::zeros({2, num, 6}, opts); // rhs vector for pose
    torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts); // hessian for cross pose depth
    torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts); // hessian cross block transpose
    torch::Tensor Cii = torch::zeros({num, ht*wd}, opts); // hessian block for depth
    torch::Tensor wi = torch::zeros({num, ht*wd}, opts); // confidence weight for BA

    for (int itr=0; itr<iterations; itr++) {

        projective_transform_kernel<<<num, THREADS>>>(
                targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
                poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
                vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());


        // build the optimization problem
        SparseBlock A(t1 - t0, 6);

        // we build motion part only B for lhs et v for rhs

        // left hand side with hessian
        A.update_lhs(Hs.reshape({-1, 6, 6}), 
                torch::cat({ii, ii, jj, jj}) - t0, 
                torch::cat({ii, jj, ii, jj}) - t0);

        // right hand side with vs
        A.update_rhs(vs.reshape({-1, 6}), 
                torch::cat({ii, jj}) - t0);

        // only at the end to get non kf poses
        if (motion_only) {
            dx = A.solve(lm, ep);

            // update poses
            pose_retr_kernel<<<1, THREADS>>>(
                    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);
        }

        else {
            // add depth residual if there are depth sensor measurements
            //const float alpha = 0.05;
            const float alpha = 0.0005;
            //printf("====== alpha 0.0005 \n");
            torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd}); // mask sur les valeurs de disparites
            torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd}); // matrice de covariance final matrix of disp
            torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd}); // confidence weight
            torch::Tensor Q = 1.0 / C; // inverse matrix depth

            torch::Tensor Ei = accum_cuda(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
            torch::Tensor E = torch::cat({Ei, Eij}, 0);

            // build schur block for the depth information using E and Q
            // we have the lhs EQEt et rhs EQw
            SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1); // schur problem

            // solve for dx for the update pose
            // A - S using dedicated operator of SparseBlock provides (B - EQEt)(v - EQw) which gives the update for the pose dx 
            dx = (A - S).solve(lm, ep); // get update pose

            torch::Tensor ix = jj_exp - t0;
            torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

            EvT6x1_kernel<<<ix.size(0), THREADS>>>(
                    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                    dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

            // solve for dz for the update depth
            dz = Q * (w - accum_cuda(dw, ii_exp, kx)); // update depth

            // update poses
            pose_retr_kernel<<<1, THREADS>>>(
                    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);

            // update disparity maps
            disp_retr_kernel<<<kx.size(0), THREADS>>>(
                    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                    dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                    kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
        }

    }

    // return updates
    return {dx, dz};
}





torch::Tensor frame_distance_cuda(
        torch::Tensor poses,
        torch::Tensor disps,
        torch::Tensor intrinsics,
        torch::Tensor ii,
        torch::Tensor jj,
        const float beta)
{
    auto opts = poses.options();
    const int num = ii.size(0);

    torch::Tensor dist = torch::zeros({num}, opts);

    frame_distance_kernel<<<num, THREADS>>>(
            poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
            ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
            jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
            dist.packed_accessor32<float,1,torch::RestrictPtrTraits>(), beta);

    return dist;
}






torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh)
{
  const int num = ix.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor counter = torch::zeros({num, ht, wd}, disps.options());

  dim3 blocks(num, 6, NUM_BLOCKS(ht * wd));

  depth_filter_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    thresh.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    counter.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  return counter;
}




torch::Tensor iproj_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics)
{

  const int nm = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  auto opts = disps.options();
  torch::Tensor points = torch::zeros({nm, ht, wd, 3}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  iproj_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return points;

}





// __global__ void projmap_kernel(
//     const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
//     const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
//     const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
//     const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
//     const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
//     torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
//     torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> valid)
// {
//
//   const int block_id = blockIdx.x;
//   const int thread_id = threadIdx.x;
//
//   const int ht = disps.size(1);
//   const int wd = disps.size(2);
//
//   __shared__ int ix;
//   __shared__ int jx;
//
//   __shared__ float fx;
//   __shared__ float fy;
//   __shared__ float cx;
//   __shared__ float cy;
//
//   __shared__ float ti[3], tj[3], tij[3];
//   __shared__ float qi[4], qj[4], qij[4];
//
//   // load intrinsics from global memory
//   if (thread_id == 0) {
//     ix = static_cast<int>(ii[block_id]);
//     jx = static_cast<int>(jj[block_id]);
//     fx = intrinsics[0];
//     fy = intrinsics[1];
//     cx = intrinsics[2];
//     cy = intrinsics[3];
//   }
//
//   __syncthreads();
//
//   // load poses from global memory
//   if (thread_id < 3) {
//     ti[thread_id] = poses[ix][thread_id];
//     tj[thread_id] = poses[jx][thread_id];
//   }
//
//   if (thread_id < 4) {
//     qi[thread_id] = poses[ix][thread_id+3];
//     qj[thread_id] = poses[jx][thread_id+3];
//   }
//
//   __syncthreads();
//
//   if (thread_id == 0) {
//     relSE3(ti, qi, tj, qj, tij, qij);
//   }
//
//   //points 
//   float Xi[4];
//   float Xj[4];
//
//   __syncthreads();
//
//   GPU_1D_KERNEL_LOOP(k, ht*wd) {
//     const int i = k / wd;
//     const int j = k % wd;
//
//     const float u = static_cast<float>(j);
//     const float v = static_cast<float>(i);
//     
//     // homogenous coordinates
//     Xi[0] = (u - cx) / fx;
//     Xi[1] = (v - cy) / fy;
//     Xi[2] = 1;
//     Xi[3] = disps[ix][i][j];
//
//     // transform homogenous point
//     actSE3(tij, qij, Xi, Xj);
//
//     coords[block_id][i][j][0] = u;
//     coords[block_id][i][j][1] = v;
//
//     if (Xj[2] > 0.01) {
//       coords[block_id][i][j][0] = fx * (Xj[0] / Xj[2]) + cx;
//       coords[block_id][i][j][1] = fy * (Xj[1] / Xj[2]) + cy;
//     }
//
//     valid[block_id][i][j][0] = (Xj[2] > MIN_DEPTH) ? 1.0 : 0.0;
//
//   }
// }




// std::vector<torch::Tensor> projmap_cuda(
//     torch::Tensor poses,
//     torch::Tensor disps,
//     torch::Tensor intrinsics,
//     torch::Tensor ii,
//     torch::Tensor jj)
// {
//   auto opts = poses.options();
//   const int num = ii.size(0);
//   const int ht = disps.size(1);
//   const int wd = disps.size(2);
//
//   torch::Tensor coords = torch::zeros({num, ht, wd, 3}, opts);
//   torch::Tensor valid = torch::zeros({num, ht, wd, 1}, opts);
//
//   projmap_kernel<<<num, THREADS>>>(
//     poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//     disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//     intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//     ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
//     jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
//     coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
//     valid.packed_accessor32<float,4,torch::RestrictPtrTraits>());
//
//   return {coords, valid};
// }



