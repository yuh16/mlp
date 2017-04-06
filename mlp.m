load('testdata2.mat');
load('mines.mat');
load('rock.mat');
load('testlabels.mat');
load('mineslabels.mat');
load('rocklabels.mat');

testdata = D2;
mines = D3';
rock = D4';

id_m0 = find(labelmin == 0);
id_m1 = find(labelmin == 1);
id_r0 = find(labelrock == 1); %revered label
id_r1 = find(labelrock == 0);

n1 = size(mines);
n2 = size(rock);

trainingdata = zeros((n1(2)+n2(2))/10,60);
train_mines = zeros(n1(2)/10,60);
train_rock = zeros(n2(2)/10,60);
k = 1;
k1 = 1;
mineslabels = zeros(1,2);
rocklabels = zeros(1,2);
for i = 10:10:n1(2)
    trainingdata(k,:) = reshape(mines(:,i-9:i),[1,60]);
    train_mines(k1,:) = reshape(mines(:,i-9:i),[1,60]);
    mineslabels(k1,:) = [1,0];
    k = k+1;
    k1 = k1+1;
end
k2 = 1;
for i = 10:10:n2(2)
    trainingdata(k,:) = reshape(rock(:,i-9:i),[1,60]);
    train_rock(k2,:) = reshape(rock(:,i-9:i),[1,60]);
    rocklabels(k2,:) = [0,1];
    k = k+1;
    k2 = k2+1;
end

testing = [train_mines(id_m1,:);train_rock(id_r1,:)];
testingLables = [ mineslabels(id_m1,:); rocklabels(id_r1,:)];

training = [train_mines(id_m0,:);train_rock(id_r0,:)];
trainingLables = [ mineslabels(id_m0,:); rocklabels(id_r0,:)];

%% training
learningRate = 0.5;
nbrOfEpochs_max = 600;
nbrOfLayers = 3;
nbrOfOutUnits = 1;
nbrOfNodesPerLayer = [60,6,2];
weights = cell(1, nbrOfLayers);
Delta_Weights = cell(1, nbrOfLayers);
%%
labels0 = zeros(length(training),2);
Samp = [training,trainingLables];
idx = randperm(length(Samp));
%%
train_data = Samp(idx,:);

labels = train_data(:,61:62);
Samples = train_data(:,1:60);

for i = 1:length(weights)-1
    weights{i} = 2*rand(nbrOfNodesPerLayer(i),nbrOfNodesPerLayer(i+1))-1 ;
    weights{i}(:,1) = 0; 
    Delta_Weights{i} = zeros(nbrOfNodesPerLayer(i), nbrOfNodesPerLayer(i+1));
end
NodesActivations = cell(1, nbrOfLayers); 
for i = 1:length(NodesActivations)
    NodesActivations{i} = zeros(1, nbrOfNodesPerLayer(i));
end
NodesBackPropagatedErrors = NodesActivations; 
mse = zeros(1,nbrOfEpochs_max);
testmse = zeros(1,nbrOfEpochs_max);
w_x1 = zeros(6,nbrOfEpochs_max);
w_x2 = zeros(6,nbrOfEpochs_max);
for ep = 1:nbrOfEpochs_max
   for spl = 1: length(Samples)
       NodesActivations{1} = Samples(spl,:);
       for Layer = 2:nbrOfLayers
           NodesActivations{Layer} = NodesActivations{Layer-1}*weights{Layer-1};
           NodesActivations{Layer} = Activation_func(NodesActivations{Layer});
       end
       gradient = Activation_func_drev(NodesActivations{nbrOfLayers});
       
       NodesBackPropagatedErrors{nbrOfLayers} = (labels(spl,:) - NodesActivations{nbrOfLayers}).*gradient;
      
       for Layer = nbrOfLayers-1:-1:1
           gradient = Activation_func_drev(NodesActivations{Layer});
           
           NodesBackPropagatedErrors{Layer}(1) =  weights{Layer}(1,:)*(NodesBackPropagatedErrors{Layer+1})';
           for node=2:length(NodesBackPropagatedErrors{Layer}) % For all the Nodes in current Layer
               NodesBackPropagatedErrors{Layer}(node) =  weights{Layer}(node,:)*(NodesBackPropagatedErrors{Layer+1}.*gradient(node))';
           end
       end
       
       for Layer = nbrOfLayers:-1:2
            
            Delta_Weights{Layer-1} =  learningRate*NodesActivations{Layer-1}'*NodesBackPropagatedErrors{Layer};
            Delta_Weights{Layer-1}(1,:) = learningRate* NodesBackPropagatedErrors{Layer}(1,:);
       end   
   
       for Layer = 1:nbrOfLayers-1
           weights{Layer} = weights{Layer} + Delta_Weights{Layer};
%            if Layer == 2
%                disp(weights{Layer}(2:4,1:2))
%            end
       end
   end
   w_x1(:,ep) = weights{2}(:,2);
   w_x2(:,ep) = weights{2}(:,1);
   [trainpred, traintruth] = test(Samples, NodesActivations, weights, labels, nbrOfLayers);
   mse(ep) = sum((traintruth - trainpred').^2/length(trainpred));
   [testpred, groundtruth] = test(testing, NodesActivations, weights, testingLables, nbrOfLayers);
   testmse(ep) = sum((testpred' - groundtruth).^2/length(testpred));
end
%%
output = movmean(mse,10);
output2 = movmean(testmse,10);
figure(1)
plot(output)
hold on
plot(output2)
hold off
xlabel('epoch')
ylabel('mean squared error')
legend('training error','testing error');

%%
figure(2)

legendInfo = cell(1);
wc = w_x1;
[nr,nc] = size(wc);
for i=1:nr
plot(1:nc,wc(i,:),'color',rand(1,3));
legendInfo{i} = ['weights ' num2str(i)]; 
hold on
end
legend(legendInfo);
hold off
xlabel('epoch');
ylabel('weights(w_i)');

%%
figure(3)

legendInfo = cell(1);
wc = w_x2;
[nr,nc] = size(wc);
for i=1:nr
plot(1:nc,wc(i,:),'color',rand(1,3));
legendInfo{i} = ['weights ' num2str(i)]; 
hold on
end
legend(legendInfo);
hold off
xlabel('epoch');
ylabel('weights(w_i)');