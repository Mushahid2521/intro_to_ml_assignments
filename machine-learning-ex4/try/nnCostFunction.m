function [J, grad] = nnCostFunction(nn_params ,...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X, y, lambda)



Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

p=hidden_layer1_size * (input_layer_size + 1);


Theta2 = reshape(nn_params((p+1):((p+hidden_layer2_size*(hidden_layer1_size+1)))), ...
                 hidden_layer2_size,(hidden_layer1_size+1));

q=((p+hidden_layer2_size*(hidden_layer1_size+1)));


Theta3=reshape(nn_params((q+1):end),num_labels,(hidden_layer2_size+1));

% Setup some useful variables
m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad=  zeros(size(Theta3));

DEL1=zeros(size(Theta1));
DEL2=zeros(size(Theta2));
DEL3=zeros(size(Theta3));


y1=zeros(num_labels,size(y,1));
y1(sub2ind(size(y1),y',1:size(y,1)))=1;
y1=y1';


X=[ones(size(X,1),1) X];
a2=sigmoid(X*Theta1');
a2=[ones(size(X,1),1) a2];
a3=sigmoid(a2*Theta2');
a3=[ones(size(X,1),1) a3];
a4=sigmoid(a3*Theta3');


J=(1/m)*sum(sum((-y1.*log(a4)-(1-y1).*log(1-a4)),2));

J=J+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+ sum(sum(Theta2(:,2:end).^2)) + sum(sum(Theta3(:,2:end).^2)));



for i=1:m
    
   b1=X(i,:);
   b2=sigmoid(Theta1*b1');
   b2=[1; b2];
   b2=b2';
   b3=sigmoid(b2*Theta2');
   b3=b3';
   b3=[1; b3];
   b3=b3';
   b4=sigmoid(Theta3*b3');
   b4=b4';
   
   del4=b4-y1(i,:);
   del3=(del4*Theta3(:,2:end)).*b3(2:end).*(1-b3(2:end));
   del2=(del3*Theta2(:,2:end)).*b2(2:end).*(1-b2(2:end));
   
   %b2(2:end).*(1-b2(2:end)) also sigmoidGradient(b2(2:end))

   DEL3=DEL3+ del4'.*b3;
   DEL2=DEL2+ del3'.*b2;
   DEL1=DEL1+ del2'.*b1;
   
 

end




%With regularization
Theta1_grad(:,1)=(1/m)*DEL1(:,1);
Theta1_grad(:,2:end)=(1/m)*DEL1(:,2:end)+(lambda/m)*Theta1(:,2:end);

%disp([size(DEL1) size(Theta1_grad)])

Theta2_grad(:,1)=(1/m)*DEL2(:,1);
Theta2_grad(:,2:end)=(1/m)*DEL2(:,2:end)+(lambda/m)*Theta2(:,2:end);

%disp([size(DEL2) size(Theta2_grad)])

Theta3_grad(:,1)=(1/m)*DEL3(:,1);
Theta3_grad(:,2:end)=(1/m)*DEL3(:,2:end)+(lambda/m)*Theta3(:,2:end);

%disp([size(DEL3) size(Theta3_grad)])





grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];



end
