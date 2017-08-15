% read in data set 
%cd matconvnet-1.0-beta23
setDir  = fullfile(toolboxdir('vision'),'visiondata','imageSets');
imds = imageDatastore('/Users/niralshah/Desktop/rcv_imgDataset/','IncludeSubfolders',true,'LabelSource',...
    'foldernames');

[trainingSet,testSet] = splitEachLabel(imds,0.5,'randomize');
true_labels = grp2idx(testSet.Labels);


%% Pre-Trained CNN 1
cd matconvnet-1.0-beta23
vl_setup();
run matlab/vl_setupnn ;
net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;
output_labels = [];
%% Go through TestSet

% Load a model and upgrade it to MatConvNet current version.
for i =1:length(testSet.Files)
    [I, fileinfo] = readimage(testSet,i);
    % Obtain and preprocess an image.
    im = I;
    
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage;

    % Run the CNN.
    res = vl_simplenn(net, im_);

    % Show the classification result.
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores);
    
 
    if best  == 760 || best == 733 % camera
        best = 1;
    elseif best == 402 % accordian
        best = 2;
    elseif best == 764 %revolver
        best = 3;
    elseif best == 509 || best == 879 || best == 811 %computer keyboard
        best = 4;
    elseif best == 964% pizza
        best = 5;
    elseif best == 777 % saxophone
        best = 6;
    elseif best  == 806 %soccer-ball
        best = 7;
    elseif best == 920 %stop-sign
       best = 8;
    elseif best == 950 %strawberry
        best =9;
    elseif best == 853 %tennis-ball
        best = 10;
    else 
        best = 11;
    end
    
    output_labels= [output_labels; best];
    %figure(1) ; clf ; imagesc(im) ;
    %title(sprintf('%s (%d), score %.3f',...
    %net.meta.classes.description{best}, best, bestScore)) ;
end 

output = [output_labels true_labels];
accuracy = 1-length(find(output_labels ~= true_labels))/length(true_labels)

%%
confusionMatrix = zeros(11,11); 
confusionMatrix(11,11) = 1;

for i= 1:length(true_labels)
     confusionMatrix(true_labels(i),output_labels(i)) = confusionMatrix(true_labels(i),output_labels(i))+1;
end

for j = 1:length(confusionMatrix)
   cum_sum = sum(confusionMatrix(j,:));
   confusionMatrix(j,:) = confusionMatrix(j,:)./cum_sum;
end

confusionMatrix
stats = confusionmatStats(confusionMatrix);
acc = stats.accuracy
sens = stats.sensitivity
specfs = stats.specificity
prec = stats.precision
recall = stats.recall

%% Pre-Trained CNN 2

vl_setup()
% setup MatConvNet
run  matlab/vl_setupnn
% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('/Users/niralshah/Desktop/imagenet-resnet-101-dag.mat')) ;
net.mode = 'test' ;
    
%%
output_labels2 = [];
for i =1:length(testSet.Files)
    % Obtain and preprocess an image.
    % load and preprocess an image
    [I, fileinfo] = readimage(testSet,i);
    im = I;
    
    im_ = single(im) ; % note: 0-255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

    % run the CNN
    net.eval({'data', im_}) ;

    % obtain the CNN otuput
    scores = net.vars(net.getVarIndex('prob')).value ;
    scores = squeeze(gather(scores)) ;

    % show the classification results
    [bestScore, best] = max(scores) ;
    if best  == 760 || best == 733 % camera
        best = 1;
    elseif best == 402 % accordian
        best = 2;
    elseif best == 764 %revolver
        best = 3;
    elseif best == 509 || best ==879 || best == 811 %computer keyboard
        best = 4;
    elseif best == 964% pizza
        best = 5;
    elseif best == 777 % saxophone
        best = 6;
    elseif best  == 806 %soccer-ball
        best = 7;
    elseif best == 920 %stop-sign
       best = 8;
    elseif best == 950 %strawberry
        best =9;
    elseif best == 853 %tennis-ball
        best = 10;
    else 
        best = 11;
    end
    
    output_labels2 = [output_labels2; best];
    %figure(1) ; clf ; imagesc(im) ;
    %title(sprintf('%s (%d), score %.3f',...
    %net.meta.classes.description{best}, best, bestScore)) ;
end 

output2 = [output_labels2 true_labels];
accuracy = 1-length(find(output_labels2 ~= true_labels))/length(true_labels)

%% Confusion Matrix:

confusionMatrix = zeros(11,11); 
confusionMatrix(11,11) = 1;

for i= 1:length(true_labels)
     confusionMatrix(true_labels(i),output_labels2(i)) = confusionMatrix(true_labels(i),output_labels2(i))+1;
end

for j = 1:length(confusionMatrix)
   cum_sum = sum(confusionMatrix(j,:));
   confusionMatrix(j,:) = confusionMatrix(j,:)./cum_sum;
end

confusionMatrix
stats = confusionmatStats(confusionMatrix);
acc = stats.accuracy
sens = stats.sensitivity
specfs = stats.specificity
prec = stats.precision
recall = stats.recall
