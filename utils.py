import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy import spatial
from statistics import median

'''
    Read images.
'''
def read_images(img_path1, img_path2):
    img1 = cv.imread(img_path1)
    img2 = cv.imread(img_path2)
    return img1, img2

'''
    Plot images.
'''
def plot_images(img_1, img_2):
    fx, plots = plt.subplots(1, 2, figsize=(20,10))
    plots[0].set_title("Target Image")
    plots[0].imshow(cv.cvtColor(img_1, cv.COLOR_BGR2RGB))

    plots[1].set_title("Template Image")
    plots[1].imshow(cv.cvtColor(img_2, cv.COLOR_BGR2RGB))

'''
    Find keypoints and descriptors with SIFT.
'''
def sift_keypoints_and_descriptors(img_1, img_2):
    sift = cv.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img_2,None)
    
    return keypoints_1, descriptors_1, keypoints_2, descriptors_2

'''
    Find keypoints and descriptors with KAZE.
'''
def kaze_keypoints_and_descriptors(img_1, img_2):
    
    gray1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)    

    detector = cv.AKAZE_create()
    
    keypoints_1, descriptors_1 = detector.detectAndCompute(gray1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(gray2, None)
    
    return  keypoints_1, descriptors_1, keypoints_2, descriptors_2

'''
    Find keypoints and descriptors with SURF.
'''
def surf_keypoints_and_descriptors(img_1, img_2):
    surf = cv.xfeatures2d.SURF_create()

    gray1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)  

    keypoints_1, descriptors_1 = surf.detectAndCompute(gray1,None)
    keypoints_2, descriptors_2 = surf.detectAndCompute(gray2,None)
    
    return keypoints_1, descriptors_1, keypoints_2, descriptors_2

'''
    Find closest matches based on distance.
'''
#cv.NORM_HAMMING for kaze
def get_best_matches(descriptors_1, desriptors_2, ratio, method=cv.NORM_L1):
    bf = cv.BFMatcher(method, crossCheck = False)
    matches = bf.knnMatch(descriptors_1,desriptors_2,k=2)

    best_matches_1 = []
    plot_best_matches_1 = []

    for m,n in matches:
        if m.distance < ratio*n.distance:
            plot_best_matches_1.append([m])
            best_matches_1.append(m)
    
    return plot_best_matches_1, best_matches_1

'''
    Draw closest matches.
'''
def draw_closest_matches(img_1, img_2, keypoints_1, keypoints_2, plot_best_matches_1):
    img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
    img = cv.drawMatchesKnn(img_1,keypoints_1,img_2,keypoints_2,plot_best_matches_1,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize = (200,200))
    plt.imshow(img)
    plt.show()

'''
    Retrieve matching point correspondences' coordinates
    for each of the images.
'''
def get_correspondences(best_matches_1, keypoints_1, keypoints_2):
    
    correspondences_1 = []
    good = []
    
    for match in best_matches_1:

        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = keypoints_1[img1_idx].pt
        (x2, y2) = keypoints_2[img2_idx].pt

        correspondences_1.append([x2, y2, x1, y1])
        
    return correspondences_1

'''
    Function to estimate the error:
        || pH - p' ||
'''

def euclidean_distance(correspondence, homographyMatrix):
    
    actualP = np.transpose([correspondence[0], correspondence[1], 1])
    actualPPrime = np.transpose([correspondence[2], correspondence[3], 1])
    
    estimatedPPrime = np.dot(homographyMatrix, actualP)
    estimatedPPrime = estimatedPPrime/estimatedPPrime[2]

    error = np.linalg.norm(actualPPrime - estimatedPPrime)

    return error

'''
    Function to estimate the homography matrix H.
'''
def estimate_homography(correspondences):
    A = []
    
    for cor in correspondences:
        p1 = [cor[0], cor[1], 1]
        p2 = [cor[2], cor[3], 1]

        A.append([-p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0, p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]])
        A.append([0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]])

    U, diagonalMatrix, Vt = np.linalg.svd(A)

    homographyMatrix = (Vt[-1]/Vt[-1][-1]).reshape((3,3))  

    return homographyMatrix, diagonalMatrix

'''
    Function to compute new Loop Number
'''
def compute_loop_number(sample_size, confidence, point_num, inlier_num):
    num = int(math.ceil(math.log10(1 - 0.01 * confidence) / math.log10(1 - float(inlier_num/point_num)**sample_size)))
    return num

'''
    Function to estimate a model via MSAC.
'''
def msac(point_correspondences, threshold=0.9, confidence=99, num_trials=1000, sample_size=4, ransac_threshold=5):
    num_pts = len(point_correspondences)
    max_dis = num_pts * threshold
    best_dis = max_dis
    best_residual = []
    H = None
    idx_trial = 1
    
    while idx_trial <= num_trials:
        
        '''
            Pick minimum set of 4 point correspondences
            and estimate a model.
        '''
        random_four_indices = random.sample(range(0, num_pts), sample_size)
        random_four = [point_correspondences[i] for i in random_four_indices]

        
        ''' 
            Estimate homography matrix H,
            based on 4 picked points.
        '''
        current_H, _ = estimate_homography(np.vstack(random_four))
        
        
        '''
            Find inliers within Euclidean distance of threshold.
        '''
        inliers = []
        acc_dis = 0
        idx_pt = 1
        while (acc_dis < best_dis) and (idx_pt < num_pts):
            distance = euclidean_distance(point_correspondences[idx_pt], current_H)
            dis = min(distance, threshold)
            if distance < ransac_threshold:
                inliers.append(point_correspondences[idx_pt])
            
            acc_dis = acc_dis + dis
            idx_pt = idx_pt + 1

        ''' 
            Always update current model,
            when a better one with more inliers
            is available.
        '''
        if acc_dis < best_dis:
            max_inliers = inliers
            best_dis = acc_dis
            H = current_H
            inlier_num = num_pts - best_dis / threshold
            num = compute_loop_number(sample_size, confidence, num_pts, inlier_num)
            num_trials = min(num_trials, num)

        idx_trial = idx_trial + 1
        
    return H, max_inliers

'''
    Function to estimate all models until #min_correspondences (used for estimation) is reached.
'''
def sequential_msac(correspondences, threshold, min_correspondences_count,confidence=99,num_trials=1000, ransac_threshold=5):
    
    models = []
    inliers = []
    epochs = 0
    while (len(correspondences) > min_correspondences_count):
        H, inliers = msac(correspondences, threshold, confidence, num_trials,min_correspondences_count, ransac_threshold)
        if len(inliers)/len(correspondences) > 0.05:
            models.append([H, inliers])
            correspondences = [i for i in correspondences if i not in inliers]
            print(len(correspondences))
            if len(correspondences) < min_correspondences_count:
                break
            epochs = 0
        else:
            if epochs == len(correspondences):
                models.append([H, inliers])
                break
            else:
                epochs += 1
    return models

'''
    Function to draw the bounding boxes
'''
def draw_bounding_boxes(img_target_plot, img_template_plot, models, scale_factor=1, isPostJLinkage=False, color=(255, 255, 255), name=None):

    plt.figure(figsize=(30,15))
    plt.title("All objects identified")
    # font
    font = cv.FONT_HERSHEY_SIMPLEX

    h,w = cv.cvtColor(img_template_plot, cv.COLOR_BGR2GRAY).shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    #color = (255, 255, 255)

    thickness = 100

    for model in models:
        if isPostJLinkage:
            H,_ = estimate_homography(np.vstack(model))
        else:
            H = model[0]
        #if abs(np.linalg.det(H)) < 0.75:
        H = scale_homography(H, scale_factor)
        dst = cv.perspectiveTransform(pts,H)
        if name:
            x,y,w,h = dst
            _x = x[0][0]
            _y = x[0][1]
            img_target_plot = cv.putText(img_target_plot, name, (_x,_y), font, 2, (255,255,255), 5, cv.LINE_AA)
        img_target_plot = cv.polylines(img_target_plot,[np.int32(dst)],True,color,5, cv.LINE_AA)
        #else:
            #break



    plt.imshow(cv.cvtColor(img_target_plot, cv.COLOR_BGR2RGB))


''' 
    Function to scale homography
'''
def scale_homography(H, scale_factor):
    return H @ [[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]]


'''
    Function to compute preference sets
'''
def get_preference_sets(correspondences, models):
    CORRESPONDENCES = 1
    BELONGS = 1
    NOT_BELONGS = 0
    
    PS = []
    for c in correspondences:
        point_PS = []
        for m in models:
            if c in m[CORRESPONDENCES]:
                point_PS.append(BELONGS)
            else:
                point_PS.append(NOT_BELONGS)
        PS.append(point_PS)
        
    return PS

'''
    Function to plot Preference Sets
'''
def plot_scatter(PS):
    N = np.where(np.array(PS) == 1)
    x = N[1]
    y = N[0]

    plt.scatter(x, y)
    plt.title('Preference Sets')
    plt.xlabel('Models')
    plt.ylabel('Points')
    plt.show()

'''
    Function to find the union of 2 points
'''
def get_union(point1, point2):
    point1 = np.asarray(point1, np.bool)
    point2 = np.asarray(point2, np.bool)
    return np.double(np.bitwise_or(point1, point2))

'''
    Function to calculate jaccard distance
'''
def get_jaccard_distance(point_1, point_2):
    point_1 = np.asarray(point_1, np.bool)
    point_2 = np.asarray(point_2, np.bool)
    return 1 - (np.double(np.bitwise_and(point_1, point_2).sum()) / get_union(point_1, point_2).sum())

'''
    Function to create initial clusters
'''
def initial_clusters(preference_sets):
    clusters = []
    for i in range(0,len(preference_sets)):
        cluster = [preference_sets[i],i]
        clusters.append(cluster)
    return clusters

'''
    J-Linkage implementation
'''
def j_linkage(preference_sets):
    clusters = initial_clusters(preference_sets)
    min_jaccard_distance = 0.0
    index_1 = 0
    index_2 = 0

    while True:
        jaccard_distances = []
        for i in range(0, len(clusters)):
            jaccard_distances_for_i = []
            for j in range(0, len(clusters)):
                if i == j:
                    jaccard_distances_for_i.append(1.0)
                else:
                    jaccard_distances_for_i.append(get_jaccard_distance(clusters[i][0], clusters[j][0]))
            jaccard_distances.append(jaccard_distances_for_i)

        min_jaccard_distance = np.nanmin(jaccard_distances)

        
        if min_jaccard_distance != 1.0:

            (index_1, index_2) = np.unravel_index(np.array(jaccard_distances).argmin(), np.array(jaccard_distances).shape)
            clusters[index_1][0] = get_union(clusters[index_1][0],clusters[index_2][0])
            clusters[index_1] += clusters[index_2][1:]
            del clusters[index_2]
        else:
            break
     
        
    return clusters

'''
    Remove the preference set from the cluster after j_linkage is performed.
'''
def remove_pref_set(cl):
    for i in cl:
        del i[0] 
    return cl

'''
    Function to get model from cluster.
'''
def get_model(cl,correspondences1):
    mod = []
    for clust in cl:
        m = []
        for ind in range(0,len(correspondences1)):
            if ind in clust:
                m.append(correspondences1[ind])
        mod.append(m)
    return mod


'''
    Function to delete outliers with a small number cluster,
    we cannot compute homography matrix from.
'''
def delete_small_model(mod, min_num=4):
    new_mod = []
    for m in range(0,len(mod)):
        if len(mod[m]) > min_num:
            new_mod.append(mod[m])
    
    return new_mod


'''
    Function to define clusters in order to calculate the diameter of cluster.
'''
def define_clusters(cluster_model):
    clusters_ = []
    clusters_size_list = []
    for each in cluster_model:
        cluster = []
        for i in each:
            cluster.append([i[2], i[3]])
        clusters_size_list.append(len(cluster))
        clusters_.append(np.array(cluster))
    return clusters_,  np.array(clusters_size_list)


'''
    Function to compute the diameter based on convex hull. 
'''
def diameter(pts):
    # We need at least 3 points to construct the convex hull
    if pts.shape[0] <= 1:
        return 0
    if pts.shape[0] == 2:
        return ((pts[0] - pts[1])**2).sum()
    # The two points which are fruthest apart will occur as vertices of the convex hull
    hull = spatial.ConvexHull(pts)
    candidates = pts[spatial.ConvexHull(pts).vertices]
    return spatial.distance_matrix(candidates, candidates).max()


'''
    Function to compute dunn index.
    The higher the index is, the better the clustering.
'''
def dunn_index(clusters_size_list, clusters_):
    max_intracluster_dist = max(diameter(clusters_[i]) for i in range(len(clusters_)))
    min_intercluster_dist = clusters_size_list.min()
    return min_intercluster_dist / max_intracluster_dist


'''
    Function to remove unfit clusters based on 
    comparison of mean diameter of all clusters and diameter inside of 
    particular cluster.
'''

def remove_far_distances_clusters(clusters_, new_models_, min_distance=100):
    list_candidates = []
    new_test_model = []
    list_dist_per_cluster = []
    mean_distance = 0
    
    ## Calculate the mean diameter of all cluters 
    ## (Idea: the matching points in target image shouldn't be far away from each other for each cluster)
    all_diameter_cluster = [diameter(clusters_[i]) for i in range(len(clusters_))]
    AVG = sum(all_diameter_cluster) / len(all_diameter_cluster)
    MEDIAN = median(all_diameter_cluster)
    
    for k,cluster_ in enumerate(clusters_):
        candidates = cluster_[spatial.ConvexHull(cluster_).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)
    
        # get indices of candidates that are furthest apart
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

        dist = math.sqrt((candidates[j][0]-candidates[i][0])**2 + (candidates[j][1]-candidates[i][1])**2)
        list_dist_per_cluster.append(dist)
    
    for k, each_cluster_dist in enumerate(list_dist_per_cluster):
        if each_cluster_dist <= MEDIAN and each_cluster_dist <= AVG:
            new_test_model.append(new_models_[k])
            
            
    return new_test_model

'''
    Function to calculate centeroid point
'''
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return [sum_x/length, sum_y/length]


'''
    Function to merge 2 cluster which close each others
'''
def merge_cluster(clusters2, ori_model):
    centroids_ = [centeroidnp(np.array(j)) for j in clusters2]
    i = 0
    j = len(centroids_)-1
    
    while i < j:
        for k in range(j, i, -1):
            dist = math.sqrt((centroids_[k][0]-centroids_[i][0])**2 + (centroids_[k][1]-centroids_[i][1])**2)
            
            #  the threshold: if the distance between center points is less than diameter of point i merge 2 clusters
            if dist < diameter(clusters2[i]):
                clusters2[i] = np.concatenate((clusters2[i], clusters2[k]))
                ori_model[i] = np.concatenate((ori_model[i], ori_model[k]))

                del centroids_[k]
                del clusters2[k]
                del ori_model[k]
        j = len(centroids_)-1
        i += 1
    
    return clusters2, ori_model
