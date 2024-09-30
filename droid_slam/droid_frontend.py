import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph

import networkx as nx
import matplotlib.pyplot as plt

import cv2

from datetime import datetime

import depth_video
import motion_filter

import os

class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(video, net.update, max_factors=48, upsample=args.upsample)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius


    def visualize_graph(self, titre, i0=None, j0=None):
        # previous edges set(zip((i,j)))
        
        # recuperer le current graph
        i = self.graph.ii.cpu().numpy()
        j = self.graph.jj.cpu().numpy()

        if i0 is not None and j0 is not None:
            # indices
            i_1 = i0.cpu().numpy()
            j_1 = j0.cpu().numpy()

            nodes_i1 = [f'i_{x}' for x in i_1]
            nodes_j1 = [f'j_{x}' for x in j_1]
            edges_1 = set(zip(nodes_i1, nodes_j1))

            nodes_i2 = [f'i_{x}' for x in i]
            nodes_j2 = [f'j_{x}' for x in j]
            edges_2 = set(zip(nodes_i2, nodes_j2))

            edges_added = edges_2 - edges_1
            edges_removed = edges_1 - edges_2

            G = nx.Graph()

# Ajouter les arêtes du second graphe
            G.add_edges_from(edges_2)

# Définir les positions des nœuds pour les organiser en lignes (par exemple)
            pos = {}
# Positionner les nœuds i en haut
            for index, node in enumerate(nodes_i2):
                pos[node] = (i[index], 1)
# Positionner les nœuds j en bas
            for index, node in enumerate(nodes_j2):
                pos[node] = (j[index], 0)

# Ajouter les nœuds des arêtes supprimées
            for edge in edges_removed:
                node1, node2 = edge
                if node1 not in pos:
                    pos[node1] = (int(node1.split('_')[1]), 1)
                if node2 not in pos:
                    pos[node2] = (int(node2.split('_')[1]), 0)

# Identifier les nœuds supprimés
            nodes_1 = set(nodes_i1 + nodes_j1)
            nodes_2 = set(nodes_i2 + nodes_j2)
            nodes_removed = nodes_1 - nodes_2

# Dessiner le graphe avec les arêtes ajoutées et supprimées en couleur
            plt.figure()

# Dessiner les nœuds
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

# Dessiner les nœuds supprimés en rouge
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_removed, node_size=700, node_color='red')

# Dessiner les arêtes inchangées
            unchanged_edges = edges_2 - edges_added
            nx.draw_networkx_edges(G, pos, edgelist=unchanged_edges, edge_color='black')

# Dessiner les arêtes ajoutées en vert
            nx.draw_networkx_edges(G, pos, edgelist=edges_added, edge_color='green')

# Dessiner les arêtes supprimées en rouge
            nx.draw_networkx_edges(G, pos, edgelist=edges_removed, edge_color='red', style='dashed')

# Ajouter les étiquettes des nœuds
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Ajouter un titre
            plt.title(titre)

# Générer un timestamp pour le nom du fichier
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
# Sauvegarder l'image dans le dossier avec un timestamp
            save_path = os.path.join("graph_images", f'graph_diff_{timestamp}.png')
            plt.savefig(save_path)
# Afficher le graphe
            #plt.show()

        else:
# Créer un graphe vide
            G = nx.Graph()
# Ajouter des nœuds avec des préfixes pour distinguer les indices i et j
            nodes_i = [f'i_{x}' for x in i]
            nodes_j = [f'j_{x}' for x in j]
# Ajouter des arêtes à partir des indices i et j
            edges = zip(nodes_i, nodes_j)
            G.add_edges_from(edges)
# Définir les positions des nœuds pour les organiser en lignes
            pos = {}
# Positionner les nœuds i en haut
            for index, node in enumerate(nodes_i):
                #pos[node] = (index, 1)
                pos[node] = (i[index], 1)
# Positionner les nœuds j en bas
            for index, node in enumerate(nodes_j):
                #pos[node] = (index, 0)
                pos[node] = (j[index], 0)
# Dessiner le graphe

            fig, ax = plt.subplots()
            nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
# Ajouter un titre
            ax.set_title(titre)
# Générer un timestamp pour le nom du fichier
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
# Sauvegarder l'image dans le dossier avec un timestamp
            save_path = os.path.join("graph_images", f'graph_diff_{timestamp}.png')
            plt.savefig(save_path)
# Afficher le graphe
            #plt.show()





    def __update(self):
        """ add edges, perform update """
        #print("+++++ update DroidFrontend")

        import pdb; pdb.set_trace()

        self.count += 1
        self.t1 += 1

        print("self.graph.ii : \n", self.graph.ii)
        print("self.graph.jj : \n", self.graph.jj)

        frame_info = "Frame " + str(self.count) + " Keyframes " + str(self.t1 - 1) + " "
        
        #self.visualize_graph(frame_info+"UPDATE - Graph")
        #self.visualize_projection(frame_info+"UPDATE - Graph")

        # keep graph before update
        ii_0 = self.graph.ii.clone()
        jj_0 = self.graph.jj.clone()

        #print("+++++ manage factor graph")
        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)
            if torch.any((self.graph.age>self.max_age) != 0):
                #self.visualize_graph(frame_info+"UPDATE - Graph post rm_factors", ii_0, jj_0)
                #self.visualize_projection(frame_info+"UPDATE - Graph post rm_factors")
                # keep graph before update
                ii_0 = self.graph.ii.clone()
                jj_0 = self.graph.jj.clone()

        print("self.graph.ii : \n", self.graph.ii)
        print("self.graph.jj : \n", self.graph.jj)


        # update graph with proximity factors init target and weight based on initial guess of pose and disp of previous iteration using video.reproject
        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        print("self.graph.ii : \n", self.graph.ii)
        print("self.graph.jj : \n", self.graph.jj)

        #self.visualize_graph(frame_info+"UPDATE - Graph post add_proximity_factors", ii_0, jj_0)
        #self.visualize_projection(frame_info+"UPDATE - Graph post add_proximity_factors")

        # keep graph before update
        ii_0 = self.graph.ii.clone()
        jj_0 = self.graph.jj.clone()

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        # print("+++++ update operator for disparity with 4 iterations to be more accurate")
        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        #self.visualize_graph(frame_info+"UPDATE - Graph post BA", ii_0, jj_0)
        #self.visualize_projection(frame_info+"UPDATE - Graph post BA")

        # keep graph before update
        ii_0 = self.graph.ii.clone()
        jj_0 = self.graph.jj.clone()

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)
        print("frontend.__update t0 ", self.t1-3, " t1 ", self.t1-2)
        print("d.shape : ", d.shape)

        # not enoough motion previous can be replaced by current t1 - 1
        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            
            #self.visualize_graph(frame_info+"UPDATE - Graph post rm_keyframe", ii_0, jj_0)

            # keep graph before update
            ii_0 = self.graph.ii.clone()
            jj_0 = self.graph.jj.clone()
           
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                #print("+++++ update operator for disparity with 2 iterations to be even more accurate")
                self.graph.update(None, None, use_inactive=True)

                #self.visualize_graph(frame_info+"UPDATE - Graph post second BA for new KF", ii_0, jj_0)

                # keep graph before update
                ii_0 = self.graph.ii.clone()
                jj_0 = self.graph.jj.clone()


        #self.visualize_graph(frame_info+"UPDATE - Graph post BA", ii_0, jj_0)
        #self.visualize_projection(frame_info+"UPDATE - End")

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True



    def visualize_projection(self, titre="Titre"):
        # get first edge
        a = self.graph.ii[0]
        b = self.graph.jj[0]

        # image resize
        # Calculer les nouvelles dimensions (1/8 de la taille originale)
        nouvelle_largeur = self.video.images.shape[3] // 8
        nouvelle_hauteur = self.video.images.shape[2] // 8

        image_a = self.video.images[a].cpu().permute(1,2,0).numpy()
        image_b = self.video.images[b].cpu().permute(1,2,0).numpy()

        # # Redimensionner l'image en utilisant cv2.resize
        image_a_resize = cv2.resize(image_a, (nouvelle_largeur, nouvelle_hauteur), interpolation=cv2.INTER_AREA)
        image_b_resize = cv2.resize(image_b, (nouvelle_largeur, nouvelle_hauteur), interpolation=cv2.INTER_AREA)
            
        # target first edge
        target_ab = self.graph.target.view(-1, nouvelle_hauteur, nouvelle_largeur, 2).permute(0,3,1,2).contiguous()[0].cpu().numpy()

        #print("target_ab.shape ", target_ab.shape)

# Affichage avec matplotlib
        #fig, ax = plt.subplots(1, 2, figsize=(12, 6))



        fig, ax = plt.subplots(figsize=(12, 6))
# Affichage des deux images
        # ax.imshow(np.vstack([np.hstack([image_a_resize, np.zeros_like(image_b_resize)]), 
        #                      np.hstack([np.zeros_like(image_a_resize), image_b_resize])]))


        ax.imshow(np.hstack([image_a_resize, np.zeros_like(image_b_resize), image_b_resize])) 

# Calculer la position relative des images dans le grand subplot
        #offset_x = image_a_resize.shape[1]
        offset_x = 2*image_a_resize.shape[1]
        #offset_y = image_a_resize.shape[0]
        offset_y = 0

        y_coords, x_coords = np.meshgrid(np.linspace(0, nouvelle_hauteur-1, nouvelle_hauteur), 
                                         np.linspace(0, nouvelle_largeur-1, nouvelle_largeur), indexing='ij')
        

        #ax[0].plot(x_coords, y_coords, 'go', markersize=2)  # cible sur Image B


# Aplatir les matrices pour une manipulation facile
        target_x, target_y = target_ab
        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()
        target_x_flat = target_x.flatten()
        target_y_flat = target_y.flatten()

        ids = np.linspace(0,nouvelle_largeur*nouvelle_hauteur-1,10, dtype=int)

        x_2d = np.array([x_coords_flat, target_x_flat + offset_x])
        y_2d = np.array([y_coords_flat, target_y_flat + offset_y])

        print(x_2d.shape)
        print(y_2d.shape)

        x2d_sampled = x_2d[:,ids]
        y2d_sampled = y_2d[:,ids]

        print(x2d_sampled.shape)
        print(y2d_sampled.shape)

# Tracer le point et les lignes
# Pixel vert dans image_a
        ax.plot(x_coords, y_coords, 'go', markersize=2)

# Pixel rouge dans image_b
        ax.plot(target_x + offset_x, target_y + offset_y, 'ro', markersize=2)

# Tracer une ligne entre les pixels de image_a et imageax.plot([x_coords_flat, target_x_flat + offset_x], [y_coords_flat, target_y_flat + offset_y], 'b-', lw=0.5)
        ax.plot(x2d_sampled, y2d_sampled, 'b-', lw=1.0)
# # Ajuster les limites des axes pour une meilleure visibilité
#         ax.set_xlim(0, image_a_resize.shape[1] + image_b_resize.shape[1])
#         ax.set_ylim(image_a_resize.shape[0] + image_b_resize.shape[0], 0)

        ax.set_title(titre)

# # Image A
#         ax[0].imshow(image_a_resize)
#         ax[0].set_title("Image A")
#
# # Image B avec les cibles
#         ax[1].imshow(image_b_resize)
#         ax[1].set_title("Image B avec Cibles")
#
#
#         #print("target_x.shape ", target_x.shape)
#         ax[1].plot(target_x, target_y, 'ro', markersize=2)  # cible sur Image B
#
#
# # Création du meshgrid pour les coordonnées de la première image
#         y_coords, x_coords = np.meshgrid(np.linspace(0, nouvelle_hauteur-1, nouvelle_hauteur), 
#                                          np.linspace(0, nouvelle_largeur-1, nouvelle_largeur), indexing='ij')
#         
#
#         ax[0].plot(x_coords, y_coords, 'go', markersize=2)  # cible sur Image B
#
#
# # Aplatir les matrices pour une manipulation facile
#         x_coords_flat = x_coords.flatten()
#         y_coords_flat = y_coords.flatten()
#         target_x_flat = target_x.flatten()
#         target_y_flat = target_y.flatten()
#         
#         #ax[1].plot([x_coords_flat, target_x_flat], [y_coords_flat, target_y_flat], 'b-', lw=0.5)
# # Dessiner les lignes de projection en une seule opération
#         # for start_x, start_y, end_x, end_y in zip(x_coords_flat, y_coords_flat, target_x_flat, target_y_flat):
#         #     ax[0].plot([start_x, end_x], [start_y, end_y], 'b-', lw=0.5)
#
#         con = plt.Line2D([x_coords_flat, target_x_flat], [y_coords_flat, target_y_flat], color="blue")
#         ax[1].add_line(con)

        plt.show()


    def __initialize(self):
        """ initialize the SLAM system """

        import pdb; pdb.set_trace()

        self.t0 = 0
        self.t1 = self.video.counter.value
        
        frame_info = "Frame " + str(self.count) + " Keyframes " + str(self.t1 - 1) + " "


        # build initial target based on pose
        # initialize target and update net inp etc in add_factors for new edges for factorgraph from video

        # print("self.graph.net.shape ", self.graph.net.shape)
        # print("self.graph.inp.shape ", self.graph.inp.shape)

        last_nonzero_index = (torch.sum(self.video.fmaps.view(self.video.fmaps.shape[0], -1), dim=1) != 0).nonzero(as_tuple=False).max().item()
        print("video fmaps size : ", last_nonzero_index)

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        # Étape 1 : Identifier les images non nulles
        non_null_indices = torch.any(self.video.images.view( self.video.images.shape[0], -1) != 0, dim=1)

        # Étape 2 : Extraire les images non nulles
        non_null_images = self.video.images[non_null_indices]

        # Affichage pour vérification
        #print(non_null_images)
        print("non_null_images.shape ", non_null_images.shape)

        #self.visualize_projection(frame_info+"INIT - Graph")

        #self.visualize_graph(frame_info+"INIT - Graph")

        # BA sur le graph non optimiser
        for itr in range(8):
            # update target using delta from raft
            self.graph.update(1, use_inactive=True)

        #self.visualize_projection(frame_info+"INIT - Graph post premier BA")

        # keep graph
        ii_0 = self.graph.ii.clone()
        jj_0 = self.graph.jj.clone()

        # update graph edges based on poses optimized from update so we can add new constraint to optimized the graph edges
        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        #self.visualize_graph(frame_info+"INIT - Graph post add_proximity_factors", ii_0, jj_0)
        
        #self.visualize_projection(frame_info+"INIT - Graph post add_proximity_factors")

        # keep graph
        ii_0 = self.graph.ii.clone()
        jj_0 = self.graph.jj.clone()

        # BA sur le graph optimise
        for itr in range(8):
            # upate target using delta from raft
            self.graph.update(1, use_inactive=True)

        #self.visualize_graph(frame_info+"INIT - Graph post BA", ii_0, jj_0)

        #self.visualize_projection(frame_info+"INIT - Graph post second BA")

        # keep graph
        ii_0 = self.graph.ii.clone()
        jj_0 = self.graph.jj.clone()

        # self.video.normalize()
        # poses
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()

        # disparity
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

        #self.visualize_graph(frame_info+"INIT - Graph final post rm_factors", ii_0, jj_0)

        #self.visualize_projection(frame_info+"INIT - Graph post rm factors")

        # keep graph
        ii_0 = self.graph.ii.clone()
        jj_0 = self.graph.jj.clone()




    def __call__(self):
        """ main update """

        #print("+++++ frontend call")
        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        
