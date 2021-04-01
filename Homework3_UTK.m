clear all
close all
clc

% load UTK dataset
load faces_matrix

%% 
% visualize the first 36 faces on a grid 6x6
plot_faces = zeros(image_side*6,image_side*6);
count = 0;
for i = 1:6
    for j = 1:6
        count = count+1;
        plot_faces(1 + (i-1)*image_side: i*image_side, 1 + (j-1)*image_side:j*image_side) = reshape(faces_matrix(:,count*40),image_side,image_side);
    end
end

% the values of the pixels are not normalized
% therefore we use imagesc to visualize the faces
imagesc(plot_faces), axes('position',[0  0  1  1]), axis off, colormap gray

%% We create the training set, subtract the average face from our training set and perform SVD
n = image_side;
m = image_side;
n_train_persons = 1600; 
perm = randperm(faces_number);
faces_matrix_shuffled = faces_matrix(:,perm);
training_faces = faces_matrix_shuffled(:,1:n_train_persons);

% evalutate the average face of the training set of faces
avg_face = mean(training_faces,2);
imagesc(reshape(avg_face,n,m)), axes('position',[0  0  1  1]), axis off, colormap gray

% mean-centering data
X = training_faces - avg_face*ones(1,size(training_faces,2));

% SVD
[U,S,V] = svd(X,'econ');
%% 
% visualize the first 16 eigenfaces on a grid 4x4
plot_faces = zeros(n*4,m*4);
count = 0;
for i = 1:4
    for j = 1:4
        count = count+1;
        plot_faces(1 + (i-1)*n: i*n, 1 + (j-1)*m:j*m) = reshape(U(:,count),n,m);
    end
end

% the values of the pixels are not normalized
% therefore we use imagesc to visualize the faces
imagesc(plot_faces), axes('position',[0  0  1  1]), axis off, colormap gray
%% reconstruction of testfaces

% 10 examples of reconstruction of faces taken outside the training set
% press enter to go on
for z = 1:10 
test_face = faces_matrix_shuffled(:,n_train_persons+randi(faces_number-n_train_persons));
test_face_X = test_face - avg_face;

n_eigenfaces = [25 50 100 200 400 800 1600];

subplot(2,4,1)
imagesc(reshape(test_face,n,m)), title('Original'), axes('position',[0  0  1  1]), axis off, colormap gray


for i = 1:7
    subplot(2,4,i+1)
    r = reshape(avg_face + U(:,1:n_eigenfaces(i)) * (U(:,1:n_eigenfaces(i))' * test_face_X),n,m);
    imagesc(r), title(['Eigenfaces used = ',num2str(n_eigenfaces(i),'%d')]), axes('position',[0  0  1  1]), axis off, colormap gray   
end
pause
end
%% reconstruction of test face Maria
% convert image to matrix
marghe = imread('margherita.jpg');
marghe_new = double(imresize(rgb2gray(marghe), [n,m]));
test_face_X = reshape(marghe_new,n*m,1) - avg_face;

% number of eigenfaces used
n_eigenfaces = [25 50 100 200 400 800 1600];


% plot
subplot(2,4,1)
imagesc(marghe_new), title('Original'), axes('position',[0  0  1  1]), axis off, colormap gray

for i = 1:7
    subplot(2,4,i+1)
    r = reshape(avg_face + U(:,1:n_eigenfaces(i)) * (U(:,1:n_eigenfaces(i))' * test_face_X),n,m);
    imagesc(r), title(['Eigenfaces used = ',num2str(n_eigenfaces(i),'%d')]), axes('position',[0  0  1  1]), axis off, colormap gray
    
end

%% econstruction of test face Marco

% convert image to matrix
marco = imread('marco.jpg');
marco_new = double(imresize(rgb2gray(marco), [n,m]));
test_face_X = reshape(marco_new,n*m,1) - avg_face;

% number of eigenfaces used
n_eigenfaces = [25 50 100 200 400 800 1600];

% plot
subplot(2,4,1)
imagesc(marco_new), title('Original'), axes('position',[0  0  1  1]), axis off, colormap gray

for i = 1:7
    subplot(2,4,i+1)
    r = reshape(avg_face + U(:,1:n_eigenfaces(i)) * (U(:,1:n_eigenfaces(i))' * test_face_X),n,m);
    imagesc(r), title(['Eigenfaces used = ',num2str(n_eigenfaces(i),'%d')]), axes('position',[0  0  1  1]), axis off, colormap gray
    
end


%% plot of the singular values
singular_values = diag(S);
semilogy(singular_values,'r','LineWidth',1.5),title('Singular values') 

%% evaluating the new coordinates for the dataset
X = faces_matrix - avg_face*ones(1,size(faces_matrix,2));
[U,S,V] = svd(X,'econ'); 

new_faces_matrix = (U' * faces_matrix);

%% finding closest faces
for j = 1:10
face_chosen = randi(2000)
face = reshape(faces_matrix(:,face_chosen),n,m);

distance_from_face = zeros(2000,1);
num_components_considered = 1000; 

for i = 1:2000
distance_from_face(i) = norm(new_faces_matrix(1:num_components_considered,i) - new_faces_matrix(1:num_components_considered, face_chosen));%marghe_U(1:num_components_considered)); %
end
[out,idx] = sort(distance_from_face);

% plot the 8 faces closest to the one chosen. If the face is from this
% dataset, the first one will alwas be the same face chosen
subplot(2,4,1)
imagesc(reshape(faces_matrix(:,face_chosen),n,m)), title('Original'), axes('position',[0  0  1  1]), axis off, colormap gray
for i = 1:7
    subplot(2,4,i+1)
    r = reshape(faces_matrix(:,idx(i)),n,m);
    imagesc(r), title([num2str(i),'° closest face']), axes('position',[0  0  1  1]), axis off, colormap gray   
end
pause
end


%% import CSV and analysis of categories
data = readtable('info_faces.csv');

women_matrix = faces_matrix(:, find(data{:,2} == 1));
men_matrix = faces_matrix(:, find(data{:,2} == 0));
white_matrix = faces_matrix(:, find(data{:,3} == 0));
black_matrix = faces_matrix(:, find(data{:,3} == 1));
asian_matrix = faces_matrix(:, find(data{:,3} == 2));
indian_matrix = faces_matrix(:, find(data{:,3} == 3));
others_matrix = faces_matrix(:, find(data{:,3} == 4));
age0_matrix = faces_matrix(:, find(data{:,1} == 0));
age1_matrix = faces_matrix(:, find(data{:,1} == 1));
age2_matrix = faces_matrix(:, find(data{:,1} == 2));
age3_matrix = faces_matrix(:, find(data{:,1} == 3));

%% plotting in PCs for classification: gender

women_faces = women_matrix;
men_faces = men_matrix;

women_faces = women_faces - avg_face*ones(1,size(women_faces,2));
men_faces = men_faces - avg_face*ones(1,size(men_faces,2));

% average gender faces
subplot(1,2,1)
imagesc(reshape(mean(women_faces,2),n,m)), title('Average woman face'), axes('position',[0  0  1  1]), axis off, colormap gray
subplot(1,2,2)
imagesc(reshape(mean(men_faces,2),n,m)), title('Average man face'), axes('position',[0  0  1  1]), axis off, colormap gray

% save the distance between the centroids with different couples of PCs as
% approximate measure of how good is that couple to separate the clusters

numb_pcs = 10;
clusters_distances = zeros(numb_pcs,numb_pcs);
for pc1 = 1:numb_pcs
    for pc2 = 1:numb_pcs
        if pc1~=pc2
            PCs = [pc1 pc2];
            coords_women = U(:,PCs)'*women_faces;
            coords_men = U(:,PCs)'*men_faces;
            
            coords_centroid1 = mean(coords_women,2);
            coords_centroid2 = mean(coords_men,2);
            clusters_distances(pc1,pc2) = norm(coords_centroid1 - coords_centroid2);
        end
    end
end

% choose the two PCs that divide most the clusters
[max_row,max_column] = find(clusters_distances == max(max(clusters_distances)));

PCs = [max_row(1) max_column(1)];
coords_women = U(:,PCs)'*women_faces;
coords_men = U(:,PCs)'*men_faces;

% plots
figure(3)
plot(coords_women(1,:),coords_women(2,:),'kd','MarkerFaceColor','k'), hold on, grid on
plot(coords_men(1,:),coords_men(2,:),'r^','MarkerFaceColor','r')
title('Distribution of sex among the two best components')
xlabel(['PC',num2str(PCs(1))]), ylabel(['PC',num2str(PCs(2))])
legend('Men','Women')
set(gca,'XTick',[0],'YTick',[0]);
hold off
figure(4)
heatmap(clusters_distances)


%% plotting in PCs for classification: ethnicity

white_faces = white_matrix;
black_faces = black_matrix;
asian_faces = asian_matrix;
indian_faces = indian_matrix;
others_faces = others_matrix;

white_faces = white_faces - avg_face*ones(1,size(white_faces,2));
black_faces = black_faces - avg_face*ones(1,size(black_faces,2));
asian_faces = asian_faces - avg_face*ones(1,size(asian_faces,2));
indian_faces = indian_faces - avg_face*ones(1,size(indian_faces,2));
others_faces = others_faces - avg_face*ones(1,size(others_faces,2));

% average ethnicity faces
figure(1)
subplot(1,5,1)
imagesc(reshape(mean(white_faces,2),n,m)), title('Average white face'), axes('position',[0  0  1  1]), axis off, colormap gray
subplot(1,5,2)
imagesc(reshape(mean(black_faces,2),n,m)), title('Average black face'), axes('position',[0  0  1  1]), axis off, colormap gray
subplot(1,5,3)
imagesc(reshape(mean(asian_faces,2),n,m)), title('Average asian face'), axes('position',[0  0  1  1]), axis off, colormap gray
subplot(1,5,4)
imagesc(reshape(mean(indian_faces,2),n,m)), title('Average indian face'), axes('position',[0  0  1  1]), axis off, colormap gray
subplot(1,5,5)
imagesc(reshape(mean(others_faces,2),n,m)), title('Average others face'), axes('position',[0  0  1  1]), axis off, colormap gray


numb_pcs = 10;
clusters_distances = zeros(numb_pcs,numb_pcs);
for pc1 = 1:numb_pcs
    for pc2 = 1:numb_pcs 
        if pc1~=pc2
            PCs = [pc1 pc2];
            coords_white = U(:,PCs)'*white_faces;
            coords_black = U(:,PCs)'*black_faces;
            coords_asian = U(:,PCs)'*asian_faces;
            coords_indian = U(:,PCs)'*asian_faces;
            coords_others = U(:,PCs)'*others_faces;
        
            coords_centroid1 = mean(coords_white,2);
            coords_centroid2 = mean(coords_black,2);
            coords_centroid3 = mean(coords_asian,2);
            coords_centroid4 = mean(coords_indian,2);
            coords_centroid5 = mean(coords_others,2);
        
            dist12 = norm(coords_centroid1 - coords_centroid2);
            dist13 = norm(coords_centroid1 - coords_centroid3);
            dist14 = norm(coords_centroid1 - coords_centroid4);
            dist15 = norm(coords_centroid1 - coords_centroid5);
            dist23 = norm(coords_centroid2 - coords_centroid3);
            dist24 = norm(coords_centroid2 - coords_centroid4);
            dist25 = norm(coords_centroid2 - coords_centroid5);
            dist34 = norm(coords_centroid3 - coords_centroid4);
            dist35 = norm(coords_centroid3 - coords_centroid5);
            dist45 = norm(coords_centroid4 - coords_centroid5);
            
            % mean distance between centroids
            clusters_distances(pc1,pc2) = mean([dist12,dist13,dist14,dist15,dist23,dist24,dist25,dist34,dist35,dist45]);
        
        end
    end
end

% choose the two PCs that divide most the clusters
[max_row,max_column] = find(clusters_distances == max(max(clusters_distances)));

PCs = [max_row(1) max_column(1)];
coords_white = U(:,PCs)'*white_faces;
coords_black = U(:,PCs)'*black_faces;
coords_asian = U(:,PCs)'*asian_faces;
coords_indian = U(:,PCs)'*indian_faces;
coords_others = U(:,PCs)'*others_faces;

% plot
figure(6)
plot(coords_white(1,:),coords_white(2,:),'b^','MarkerFaceColor','b'), hold on, grid on
plot(coords_black(1,:),coords_black(2,:),'r^','MarkerFaceColor','r')
plot(coords_asian(1,:),coords_asian(2,:),'y^','MarkerFaceColor','y')
plot(coords_indian(1,:),coords_indian(2,:),'g^','MarkerFaceColor','g')
plot(coords_others(1,:),coords_others(2,:),'k^','MarkerFaceColor','k')

title('Distribution of ethnicity among the two best components')
xlabel(['PC',num2str(PCs(1))]), ylabel(['PC',num2str(PCs(2))])
legend('White','Black','Asian','Indian','Others')
set(gca,'XTick',[0],'YTick',[0]);
hold off
figure(7)
heatmap(clusters_distances)


%% plotting in PCs for classification: ages

age0_faces = age0_matrix;
age1_faces = age1_matrix;
age2_faces = age2_matrix;
age3_faces = age3_matrix;

age0_faces = age0_faces - avg_face*ones(1,size(age0_faces,2));
age1_faces = age1_faces - avg_face*ones(1,size(age1_faces,2));
age2_faces = age2_faces - avg_face*ones(1,size(age2_faces,2));
age3_faces = age3_faces - avg_face*ones(1,size(age3_faces,2));

% average age faces
subplot(1,4,1)
imagesc(reshape(mean(age0_faces,2),n,m)), title('Average child face'), axes('position',[0  0  1  1]), axis off, colormap gray
subplot(1,4,2)
imagesc(reshape(mean(age1_faces,2),n,m)), title('Average young face'), axes('position',[0  0  1  1]), axis off, colormap gray
subplot(1,4,3)
imagesc(reshape(mean(age2_faces,2),n,m)), title('Average middle-aged face'), axes('position',[0  0  1  1]), axis off, colormap gray
subplot(1,4,4)
imagesc(reshape(mean(age3_faces,2),n,m)), title('Average old face'), axes('position',[0  0  1  1]), axis off, colormap gray


numb_pcs = 10;
clusters_distances = zeros(numb_pcs,numb_pcs);
for pc1 = 1:numb_pcs
    for pc2 = 1:numb_pcs 
        if pc1~=pc2
            PCs = [pc1 pc2];
            coords_age0 = U(:,PCs)'*age0_faces;
            coords_age1 = U(:,PCs)'*age1_faces;
            coords_age2 = U(:,PCs)'*age2_faces;
            coords_age3 = U(:,PCs)'*age3_faces;
        
            coords_centroid1 = mean(coords_age0,2);
            coords_centroid2 = mean(coords_age1,2);
            coords_centroid3 = mean(coords_age2,2);
            coords_centroid4 = mean(coords_age3,2);
        
            dist12 = norm(coords_centroid1 - coords_centroid2);
            dist13 = norm(coords_centroid1 - coords_centroid3);
            dist14 = norm(coords_centroid1 - coords_centroid4);
            dist23 = norm(coords_centroid2 - coords_centroid3);
            dist24 = norm(coords_centroid2 - coords_centroid4);
            dist34 = norm(coords_centroid3 - coords_centroid4);
            
            % mean distance between centroids
            clusters_distances(pc1,pc2) = mean(dist12+dist13+dist14+dist23+dist24+dist34, 2);
        
        end
    end
end

% choose the two PCs that divide most the clusters
[max_row,max_column] = find(clusters_distances == max(max(clusters_distances)));

PCs = [max_row(1) max_column(1)];
coords_age0 = U(:,PCs)'*age0_faces;
coords_age1 = U(:,PCs)'*age1_faces;
coords_age2 = U(:,PCs)'*age2_faces;
coords_age3 = U(:,PCs)'*age3_faces;

% plot
figure(5)
plot(coords_age0(1,:),coords_age0(2,:),'b^','MarkerFaceColor','b'), hold on, grid on
plot(coords_age1(1,:),coords_age1(2,:),'r^','MarkerFaceColor','r')
plot(coords_age2(1,:),coords_age2(2,:),'y^','MarkerFaceColor','y')
plot(coords_age3(1,:),coords_age3(2,:),'g^','MarkerFaceColor','g')
title('Distribution of age among the two best components')
xlabel(['PC',num2str(PCs(1))]), ylabel(['PC',num2str(PCs(2))])
legend('Child','Young','Middle-aged','Old')
set(gca,'XTick',[0],'YTick',[0]);
hold off
figure(6)
heatmap(clusters_distances)

%% kNN on dataset projected on the first eigenfaces: age

% set best dimension and best k
loss1_k = zeros(8,1);
loss1 = zeros(18,1);
ind1_k = zeros(18,1);
target1 = data{:,1};

for n = 3:20
    for k = 1:2:15
        data1 = new_faces_matrix(:,1:n);
        Md1 = fitcknn(data1,target1,'NumNeighbors',k);
        CVMdl1 = crossval(Md1);
        cvmdlloss1 = kfoldLoss(CVMdl1);
        loss1_k((k+1)/2) = cvmdlloss1;
    end
    [loss1(n-2), ind1_k(n-2)] = min(loss1_k); 
end

n_values = 1:18;
figure()
plot(n_values,loss1,'-o'),title('Dimension')
xlabel('n'),ylabel('Loss'),xticks(values)

% retrieve best k for best n
[min_loss1, best1_n] = min(loss1);
k_values = 1:2:15;
ind1_best_k = ind1_k(best1_n);
best1_k = k_values(ind1_best_k);


%% kNN on dataset projected on the first eigenfaces: gender

% set best dimension and best k
loss2_k = zeros(8,1);
loss2 = zeros(18,1);
ind2_k = zeros(18,1);
target2 = data{:,2};

for n = 3:20
    for k = 1:2:15
        data2 = new_faces_matrix(:,1:n);
        Md2 = fitcknn(data2,target2,'NumNeighbors',k);
        CVMdl2 = crossval(Md2);
        cvmdlloss2 = kfoldLoss(CVMdl2);
        loss2_k((k+1)/2) = cvmdlloss2;
    end
    [loss2(n-2), ind2_k(n-2)] = min(loss2_k); 
end

n_values = 1:18;
figure()
plot(n_values,loss1,'-o'),title('Dimension')
xlabel('n'),ylabel('Loss'),xticks(values)

% retrieve best k for best n
[min_loss2, best2_n] = min(loss2);
k_values = 1:2:15;
ind2_best_k = ind2_k(best2_n);
best2_k = k_values(ind2_best_k);


%% kNN on dataset projected on the first eigenfaces: race

% set best dimension and best k
loss3_k = zeros(8,1);
loss3 = zeros(18,1);
ind3_k = zeros(18,1);
target3 = data{:,3};

for n = 3:20
    for k = 1:2:15
        data3 = new_faces_matrix(:,1:n);
        Md3 = fitcknn(data3,target3,'NumNeighbors',k);
        CVMdl3 = crossval(Md3);
        cvmdlloss3 = kfoldLoss(CVMdl3);
        loss3_k((k+1)/2) = cvmdlloss3;
    end
    [loss3(n-2), ind3_k(n-2)] = min(loss3_k); 
end

n_values = 1:18;
figure()
plot(n_values,loss3,'-o'),title('Dimension')
xlabel('n'),ylabel('Loss'),xticks(values)

% retrieve best k for best n
[min_loss3, best3_n] = min(loss3);
k_values = 1:2:15;
ind3_best_k = ind3_k(best3_n);
best3_k = k_values(ind3_best_k);







