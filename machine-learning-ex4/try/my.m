input_layer_size  = 400;
hidden_layer1_size = 25;   
hidden_layer2_size=25;
num_labels = 10;

load('ex4data1.mat');
m = size(X, 1);


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3=  randInitializeWeights(hidden_layer2_size,num_labels);


initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];


options = optimset('GradObj','on','MaxIter', 50);

lambda = 1;

costFunction = @(p) nnCostFunction(p,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,X,y,lambda);
                               
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

p=hidden_layer1_size * (input_layer_size + 1);


Theta2 = reshape(nn_params((p+1):((p+hidden_layer2_size*(hidden_layer1_size+1)))), ...
                 hidden_layer2_size,(hidden_layer1_size+1));

q=((p+hidden_layer2_size*(hidden_layer1_size+1)));


Theta3=reshape(nn_params((q+1):end),num_labels,(hidden_layer2_size+1));

pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
