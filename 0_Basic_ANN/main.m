%% Implementation of a basic neural network with backpropagation
% %The main reference for this is 
% @article{berg2017unified,
%   title={A unified deep artificial neural network approach to partial differential equations in complex geometries},
%   author={Berg, Jens and Nystr{\"o}m, Kaj},
%   journal={arXiv preprint arXiv:1711.06464},
%   year={2017}
% }
% The backpropagation algorithm is described in it in section 2. 


%RUNNING THE CODE
    % Press F5. You get two plots, one is a plot of the function to be
    % approximated (updated each hundred iterations) and one is the value
    % of the cost (error) function gradient. When gradient is close to
    % zero, convergence achieved. 
%CONCEPT
    % Neural network-sequence of layers comprised of nodes. Each node connected
    % by weight multiplier to each neuron in succeeding and preceding layer. 
%QUANTITIES
% Two ingredients are needed:
    % inputs & outputs: for function fitting, these are needed as training
    % data. 
    % weights & biases: both cell arrays, indexed 1 to L (L-number of layers),
        % weights contains matrices connecting each layer to the one preceding it.
        % biases contains vectors that are the biases for each layer. 
        % note: the first elements of each cell array are empty or are
    % placeholders (since they correspond to inputs). There for nicer
    % indexing.
%FUNCTIONS
    % Forward pass: computing output of the network. 
    % Backward pass: The "meat". How we get the derivative of the cost function
        % w.r.t each parameter of the network. This implements the formulas in
        % the reference.
        % Main intermediate quantity is input to neuron, as it contains all the
        % information on its "configuration" - the bias and the weights going
        % into it. The derivatives w.r.t the weights and biases of the neuron
        % are calculated from it, using chain rule. 
    % Sigmoid & sigmoid prime are sigmoid and its derivative. This is the
    % nonlinearity used here for all intermediate layers. 

clear all; clc; close all
inputs=[];
outputs=[];
grad_tol=1e-7;
%grad_C=sum(y_l{end}-outputs);
gradient_fn = @(y_L, y_act) sum(y_L-y_act);

% mini_batch_size: For each iteration, how many random samples we take to approximate the gradient. 
% The gradient is a function of all the inputs/outputs, its expensive to
% calculate using all of them for each iteration, so we take a sample. 
% Small value - gradient noisy, but we go fast. Too big value: gradient
% less noisy, but expensive for each iteration.
mini_batch_size=20;

%maximum number iterations
max_iter=10000;

learning_rate=1; %gradient descent step size
momentum_rate=0.1; %implement momentum. At each gradient descent step, we add the previous step with a multiplier to help with local minima
inputs=[linspace(1,5,250)];%; linspace(1,5,10)];  %INPUTS
outputs=[arrayfun(@(x) sin(x), inputs(1,:))];%; arrayfun(@(x) cos(x), inputs)]; %OUTPUTS

hidden_config=[4]; %hidden layer sizes, give an array of integers. 

sz=size(inputs); in_dim=sz(1);
sz=size(outputs); out_dim=sz(1);

layer_sizes=[in_dim, hidden_config, out_dim]; 

%% Weights and biases initialization

%random weight and bias intialization
weights=cell(1,length(layer_sizes)); 
biases=cell(1, length(layer_sizes)); 
for l=2:length(weights) %l for layers
    weights{l}=rand(layer_sizes(l), layer_sizes(l-1));
    biases{l}=rand(layer_sizes(l), 1);
end

biases{1}=inputs(:,1); %as a placeholder, this is used for indexing,


for iter=1:max_iter
    sample_idx=floor(rand(1,mini_batch_size)*length(inputs))+1; %pick a mini_batch
    
    grad_C=zeros(size(biases{end})); %gradient of cost function w.r.t. net output
    for i=1:sample_idx %go through minibatch, add up gradient
        if iter>1
            prev_dC_dw=dC_dw; prev_dC_db=dC_db; 
        end
       
        [y_l, z_l]=forward_pass(inputs(:,i), weights, biases);
        grad_C=sum(y_l{end}-outputs(:,i));
        [dC_dw, dC_db, grad_C_inc] = backward_pass( weights, biases, y_l, z_l, grad_C);
        
        
        if iter==1
            for j=2:length(weights)
                weights{j}=weights{j}-dC_dw{j}.*learning_rate;
                biases{j}=biases{j}-dC_db{j}.*learning_rate;        
            end
        else
            for j=2:length(weights)
                weights{j}=weights{j}-dC_dw{j}.*learning_rate-momentum_rate*prev_dC_dw{j}.*learning_rate;
                biases{j}=biases{j}-dC_db{j}.*learning_rate-momentum_rate*prev_dC_db{j}.*learning_rate;        
            end
            
        end
        
        
        
        
        grad_C=grad_C+grad_C_inc;
    end
    
    %grad_C=mean(grad_C); %take the mean 
    if abs(grad_C)<grad_tol
        break;
    end
    grad_C_arr(iter)=grad_C;
    figure(1)
    [~,~,net_out]=forward_pass(inputs, weights, biases);

    subplot(2,1,1)
    if rem(iter,100)==0
        plot(inputs(1,:), outputs(1,:), '-o'); hold on
        plot(inputs(1,:), net_out(1,:), '-o'); 
        legend('actual', 'net')
        hold off
    end
    subplot(2,1,2)
    
   % plot(grad_C_arr)
    if iter>100 
        plot((iter-100):iter, grad_C_arr((iter-100):iter));
        legend('gradient val')
    end
end



















% 
% function [] = net_error(inputs, outputs,  weights, biases) 
%     
% end


