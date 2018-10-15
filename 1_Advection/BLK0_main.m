clear all; clc; close all
% ADVECTION SOLUTION WITH NEURAL NET
    %Taken from same paper as Basic_ANN. Replicate basic example. 
    
% %The main reference for this is 
% @article{berg2017unified,
%   title={A unified deep artificial neural network approach to partial differential equations in complex geometries},
%   author={Berg, Jens and Nystr{\"o}m, Kaj},
%   journal={arXiv preprint arXiv:1711.06464},
%   year={2017}
% }
% This is the first numerical example. The terminology refers to the
% notation in that paper, especially section 3.3.1


x_dom=linspace(0,1,100); 
%Analytical Solution
% exact_sol=@(x) sin(2*pi*x)*cos(4*pi*x)+1;
% plot(x_dom, arrayfun(exact_sol, x_dom))


global G; G=@(x) 1; %boundary extension function 
global D; D=@(x) x; %distance function

in_dim=1; out_dim=1; hidden_config=[4]; 
layer_sizes=[in_dim, hidden_config, out_dim]; 

%% Weights and biases initialization

%random weight and bias intialization
weights=cell(1,length(layer_sizes)); 
biases=cell(1, length(layer_sizes)); 
for l=2:length(weights) %l for layers
    weights{l}=rand(layer_sizes(l), layer_sizes(l-1));
    biases{l}=rand(layer_sizes(l), 1);
end
biases{1}=x_dom(:,1); %as a placeholder, this is used for indexing,

i=10;


% Writing gradient of cost function (residual), need (ref to p. 18 in
% reference):
%Network output y_L
[y,z,~]=forward_pass(x_dom(:, i), weights, biases);
L=length(weights); 
    
%Derivative network output w.r.t. network parameters del y_1_L del p
%with p being network parameter. (p. 11 in reference)
%slight change from Basic_ANN. Pass the gradient value instead of function. Cost fn identitiy here
grad_val=y{L}; 
[dy_dw, dy_db, ~] = backward_pass(x_dom(:, i), weights, biases, grad_val);
%Second degree partial derivative: (d^2 y) over (d x_i dp) with p being
%network parameter. (p. 13 in reference). Differentiate backprop alg
%values above. Need:
    %Change of y w.r.t. network input
        %Separate algorithm 




%Compute 




%%





close all;


