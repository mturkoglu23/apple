clc;clear;
net = alexnet;
net1 = densenet201;
net2 = googlenet;
layer= 'fc6';
layer1 ='fc1000';
layer2= 'pool5-drop_7x7_s1';

imds = imageDatastore('......','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,idmsTest] = splitEachLabel(imds,0.75,'randomized');

uzunluk=numel(imdsTrain.Labels);

for i=1:uzunluk
   img=readimage(imdsTrain,i);
   img1=imresize(img,[227 227]);
   img=imresize(img,[224 224]);

   alex_feat(:,1)  = activations(net,img1,layer);
   dense_feat(:,1) = activations(net1,img,layer1);
   google_feat(:,1) = activations(net2,img,layer2);
   
   alex_Train{i}=alex_feat;
   dense_Train{i}=dense_feat;
   google_Train{i}=google_feat;
end
Train_labels=imdsTrain.Labels;
 

 uzunluk=numel(idmsTest.Labels);

 for i=1:uzunluk
     i
   img=readimage(idmsTest,i);
   img1=imresize(img,[227 227]);
   img=imresize(img,[224 224]);

   alex_feat(:,1)  = activations(net,img1,layer);
   dense_feat(:,1) = activations(net1,img,layer1);
   google_feat(:,1) = activations(net2,img,layer2);
   
   alex_Test{i}=alex_feat;
   dense_Test{i}=dense_feat;
   google_Test{i}=google_feat;
 end
Test_labels=idmsTest.Labels;

%%

Trn1=alex_Train';
Test1=alex_Test';
Trn2=dense_Train';
Test2=dense_Test';
Trn3=google_Train';
Test3=google_Test';
Trn_label=categorical(Train_labels);
Tst_label=categorical(Test_labels);

inputSize = 4096;
inputSize1 = 1024;
inputSize2 = 1000;
numHiddenUnits = 1024;
numHiddenUnits1 = 300;
numClasses = 4;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

layers1 = [ ...
    sequenceInputLayer(inputSize1)
    lstmLayer(numHiddenUnits1,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

layers2 = [ ...
    sequenceInputLayer(inputSize2)
    lstmLayer(numHiddenUnits1,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

        
options = trainingOptions('sgdm',...
         'MaxEpochs',10,...  
        'MiniBatchSize',7,...
      'InitialLearnRate',1e-4,...
        'Verbose',false);

netw = trainNetwork(Trn1,Trn_label,layers,options);
netw1 = trainNetwork(Trn2,Trn_label,layers2,options);
netw2 = trainNetwork(Trn3,Trn_label,layers1,options);

YPred1 = classify(netw,Test1);
YPred2 = classify(netw1,Test2);
YPred3 = classify(netw2,Test3);

for i=1:length(YPred1)
      diz=[YPred1(i),YPred2(i),YPred3(i)];
        YPredson(i)=mode(diz);
end

accuracy = mean(YPredson' == Tst_label)

  
