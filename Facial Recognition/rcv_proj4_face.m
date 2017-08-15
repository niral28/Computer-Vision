%% RCV eigenface recognition
% Niral Shah
% 12/03/16

setDir  = fullfile(toolboxdir('vision'),'visiondata','imageSets');
imds = imageDatastore('/Users/niralshah/Desktop/yalefaces','IncludeSubfolders',true,'LabelSource',...
    'foldernames');
[trainingSet,testSet] = splitEachLabel(imds,0.7,'randomize');
training_labels = grp2idx(trainingSet.Labels);
test_labels = grp2idx(testSet.Labels);

A = []; 
average = zeros(150,150);
for i =1:length(trainingSet.Files)
    [I, fileinfo] = readimage(trainingSet,i);
    
    
       [x,y,z] = size(I);
    if(z ~= 1)
        I = rgb2gray(I) ;
        I = imresize(I,[150,150]);
    else
        I = imresize(I,[150,150]);   
    end
    average = average + double(I);
    T = reshape(I,[],1);
     A = [A T];
%     figure; 
%     imshow(I);
%     pause(2);
%     close;
end
%% Mean Face

average = average/i;

hfig= figure(2);
set(hfig,'Position',[0 0 150 150])
imagesc(average);
colormap gray;
title('Mean Face');
%%
a = reshape(average,[],1);
A_meansub = double(A)-a;

[U,S,V] = svd(A_meansub);

%% Check if training data can be reconstructed: 

img = A(:,10); % image to reconstruct 
[I, fileinfo] = readimage(trainingSet,10);

k = 12000;
Uk = U(:,1:k);
csum = zeros(length(Uk),1);

for j = 1:k
    csum = csum + (Uk(:,j)'*double(img)*Uk(:,j));
end
csum= csum+a;
new_img = reshape(csum,150,150);

hfig=figure(3);
set(hfig,'Position',[0 0 150 150])
imagesc(new_img);
colormap gray;
title(['Class: ' char(fileinfo.Label) ' ']);

%% Face Recognition (based on class label)
output = []; 
class_label = [];
labels = grp2idx(trainingSet.Labels);
true_label = grp2idx(testSet.Labels);

for i =1:length(testSet.Files)
    [I, fileinfo] = readimage(testSet,i);
    
      
       [x,y,z] = size(I);
    if(z ~= 1)
        I = rgb2gray(I) ;
        I = imresize(I,[150,150]);
    else
        I = imresize(I,[150,150]);   
    end
    
    T = double(reshape(I,[],1));
    T = T-a;
    csum = zeros(length(Uk),1);
    for l = 1:k
        csum = csum + Uk(:,l)'*double(T)*Uk(:,l);
    end
   
    [idx,D] = knnsearch(csum',A_meansub');
    [value,index] = min(D);
    [I2, fileinfo2] = readimage(trainingSet,index);
    output = [output;fileinfo.Label fileinfo2.Label];
    class_label = [class_label;labels(index)];
    new_img = reshape(csum,150,150);
    
%     hfig=figure(3);
%     set(hfig,'Position',[10 10 150 150])
%     imagesc(new_img);
%     colormap gray;
%     title(['Class:' char(fileinfo2.Label) ' ']);
%     pause(2.5);
%     close;
end
%%
numberCorrect = length(testSet.Files)-length(find(output(:,1) ~= output(:,2)));
doubleCheck = length(testSet.Files)-length(find(class_label(:) ~= true_label(:)));

acc = doubleCheck/length(testSet.Files)
accuracy = numberCorrect/length(testSet.Files);
confusionMatrix = zeros(15);

for i= 1:length(true_label)
     confusionMatrix(true_label(i),class_label(i)) = confusionMatrix(true_label(i),class_label(i))+1;
end


for j = 1:length(confusionMatrix)
   cum_sum = sum(confusionMatrix(j,:));
   confusionMatrix(j,:) = confusionMatrix(j,:)./cum_sum;
end

confusionMatrix
stats = confusionmatStats(confusionMatrix)        
% accuracy with K =10000 is 86.67 % 
% accuracy with K =1000 is 86.67%

%% Project Requirements:
%Use 5 splits of the data (50-50 split or up to 30-70).
% Provide a confusion matrix for each method in the description of results. Include
% the input and output class names in the confusion matrix. Use ConfusionMatStats
% to provide confusion matrix statistics. Write a paragraph discussing these results.
