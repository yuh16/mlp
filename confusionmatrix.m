
net = feedforwardnet(6, 'trainlm');

x = Samples';
t = labels';

x_test = testing;
testlabels = testingLables;

net = train(net, x, t);
y_train = net(x);

y_test = net(x_test');


[n1,m1] = size(y_train);
[n2,m2] = size(y_test);

groundtruth1 = zeros(1,m1);
train_pred = zeros(1,m1);

groundtruth2 = zeros(1,m2);
test_pred = zeros(1,m2);

for i = 1:m1

    if(t(1,i) > t(2,i))
       groundtruth1(i) = 1;
    end
    
    if(y_train(1,i) >y_train(2,i))
       train_pred(i) = 1;
    end
    
end


for j = 1:m2
    
    if(testlabels(j,1) > testlabels(j,2))
       groundtruth2(j) = 1;
    end
    
    if(y_test(1,j) >y_test(2,j))
       test_pred(j) = 1;
    end
end

[C1,order1] = confusionmat(groundtruth1,train_pred)
[C2,order2] = confusionmat(groundtruth2,test_pred)

