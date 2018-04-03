function [ J ] = calculateCost( X,y,theta )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
m=size(X,1);
J=0;
J=(1/(2*m))*sum((X*theta-y).^2);

end

