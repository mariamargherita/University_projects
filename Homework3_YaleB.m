clear all
close all
clc

% load YaleB dataset
load allFaces.mat


%% 
% visualize the first 36 faces on a grid 6x6
plot_faces = zeros(n*6,m*6);
count = 0;
for i = 1:6
    for j = 1:6
        count = count+1;
        plot_faces(1 + (i-1)*n: i*n, 1 + (j-1)*m:j*m) = reshape(faces(:,sum(nfaces(1:count-1))+1),n,m);
    end
end

% the values of the pixels are not normalized
% therefore we use imagesc to visualize the faces
imagesc(plot_faces), axes('position',[0  0  1  1]), axis off, colormap gray
title('Images of 36 subjects in YaleB dataset')

%% create the training set, subtract the average face from our training set and perform SVD

% training set
n_train_persons = 36;
training_faces = faces(:,1:sum(nfaces(1:n_train_persons)));

% evalutate the average face of the training set of faces
avg_face = mean(training_faces,2); % mean-centering data
imagesc(reshape(avg_face,n,m)), axes('position',[0  0  1  1]), axis off, colormap gray

% subtraction of average face from training set
X = training_faces - avg_face*ones(1,size(training_faces,2));

% SVD
[U,S,V] = svd(X,'econ'); % because I don't want a matrix 32000x32000 but 32000x2400 ('econ')
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
%% reconstruction of 37th test face with different numbers of eigenfaces

% 37th face
test_face = faces(:,sum(nfaces(1:n_train_persons))+1);
test_face_X = test_face - avg_face;

% number of eigenfaces used
n_eigenfaces = [25 50 100 200 400 800 1600];

% plot
subplot(2,4,1)
imagesc(reshape(test_face,n,m)), title('Original'), axes('position',[0  0  1  1]), axis off, colormap gray

for i = 1:7
    subplot(2,4,i+1)
    r = reshape(avg_face + U(:,1:n_eigenfaces(i)) * (U(:,1:n_eigenfaces(i))' * test_face_X),n,m);
    imagesc(r), title(['Eigenfaces used = ',num2str(n_eigenfaces(i),'%d')]), axes('position',[0  0  1  1]), axis off, colormap gray
    
end

%% reconstruction of test face Marco

% convert image to matrix and build test face
marco = imread('marco.jpg');
marco = double(imresize(rgb2gray(marco), [n,m]));
test_face_X = reshape(marco,n*m,1) - avg_face;

% number of eigenfaces used
n_eigenfaces = [25 50 100 200 400 800 1600];

% plot
subplot(2,4,1)
imagesc(marco), title('Original'), axes('position',[0  0  1  1]), axis off, colormap gray

for i = 1:7
    subplot(2,4,i+1)
    r = reshape(avg_face + U(:,1:n_eigenfaces(i)) * (U(:,1:n_eigenfaces(i))' * test_face_X),n,m);
    imagesc(r), title(['Eigenfaces used = ',num2str(n_eigenfaces(i),'%d')]), axes('position',[0  0  1  1]), axis off, colormap gray
    
end
%% Reconstruction of test face Margherita

% convert image to matrix and build test face
marghe = imread('margherita.jpg');
marghe = double(imresize(rgb2gray(marghe), [n,m]));
test_face_X = reshape(marghe,n*m,1) - avg_face;

% number of eigenfaces used
n_eigenfaces = [25 50 100 200 400 800 1600];

% plot
subplot(2,4,1)
imagesc(marghe), title('Original'), axes('position',[0  0  1  1]), axis off, colormap gray

for i = 1:7
    subplot(2,4,i+1)
    r = reshape(avg_face + U(:,1:n_eigenfaces(i)) * (U(:,1:n_eigenfaces(i))' * test_face_X),n,m);
    imagesc(r), title(['n eigenfaces=',num2str(n_eigenfaces(i),'%d')]), axes('position',[0  0  1  1]), axis off, colormap gray
    
end
%% plot of the singular values
singular_values = diag(S);
semilogy(singular_values, 'LineWidth',2),title('Singular values') 

%% plotting in PCs for classification of 5th and 10th persons

first_person = 5;
second_person = 10;

first_face = faces(:,1+sum(nfaces(1:first_person-1)):sum(nfaces(1:first_person)));
second_face = faces(:,1+sum(nfaces(1:second_person-1)):sum(nfaces(1:second_person)));

first_face = first_face - avg_face*ones(1,size(first_face,2));
second_face = second_face - avg_face*ones(1,size(second_face,2));

PCs = [5 6]; % project on 5th and 6th PCs
coords_first = U(:,PCs)'*first_face;
coords_second = U(:,PCs)'*second_face;

% plot
figure
plot(coords_first(1,:),coords_first(2,:),'kd','MarkerFaceColor','k')
axis([-4000 4000 -3000 6000]), hold on, grid on
plot(coords_second(1,:),coords_second(2,:),'r^','MarkerFaceColor','r'),
xlabel(['Principal Component ' , num2str(PCs(1))]), ylabel(['Principal Component ' , num2str(PCs(2))])
legend(['Person ', num2str(first_person)],['Person ', num2str(second_person)])
set(gca,'XTick',[0],'YTick',[0]);


%% kNN on dataset projected on the first eigenfaces

% set dimension and k
accuracy = zeros(7,1);
loss_k = zeros(5,1);
loss = zeros(18,1);
ind_k = zeros(18,1);
persons_classified = [1+sum(nfaces(1:first_person-1)):sum(nfaces(1:first_person)),1+sum(nfaces(1:second_person-1)):sum(nfaces(1:second_person))];

p = 0;
for i = 1:36
    p = p+1;
    person_index(1+sum(nfaces(1:p-1)):sum(nfaces(1:p))) = p;
end
target = person_index(persons_classified)';

data = (U(:,5:6)'*training_faces(:,persons_classified))';
ind = randperm(length(data));
train_f = data(ind(1:100),:);
test_f = data(ind(101:end),:);
train_target = target(ind(1:100));
test_target = target(ind(101:end));
%% 
    for k = 3:2:11
        Mdl = fitcknn(train_f,train_target,'NumNeighbors',k);
        pred = predict(Mdl,test_f);
        k
        c = confusionmat(test_target,pred)
        accuracy((k+1)/2-1) = sum(diag(c)) / sum(sum(c))
    end
    %[loss(count), ind_k(count)] = min(loss_k); 
%en
%% 

n_values = 1:count;
figure()
eig = 10:10:400;
plot(eig,loss,'-o'),title('k-NN performance with best k')
xlabel('Number of eigenfaces considered'),ylabel('Error rate')

% retrieve best k for best n
[min_loss, best_n] = min(loss);
k_values = 3:2:11;
ind_best_k = ind_k(best_n)
best_k = k_values(ind_best_k)

