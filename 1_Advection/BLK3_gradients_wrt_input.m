function [J_y, J_z] = BLK3_gradients_wrt_input(weights, z)
%COMPUTE GRADIENTS OF NETWORK QUANTITIES Y and Z WRT INPUT X. 
    %Refer to Appendix A advection problems, 
    %We need the gradients of all the neuron weighted inputs and outputs,
    %with respect to input x. 
    %This is done by taking the partial derivatives of the network forward
    %pass w.r.t. x_i, which defines a new neural net with the required
    %gradients. 
    
%DEFINITIONS: 
    %L: number of layers of neural network. 
%INPUTS
    %weights and biases characterize the neural network. 
        %weights: cell array, size 1 to L where L is number of layers. The l'th
            %entry contains matrix W(l) which connects the l'th layer to the l-1th
    %x: M by 1 vector, input to neural net
%OUTPUTS
    %Recall: z is the weighted input to each neuron (wy_(l-1)+b), and y is the output of
    %the neuron. y is z with the nonlinearity applied. 
    
    %J_y: Cell array of size L, with the l'th element containing a Jacobian
        %matrix of y(l) w.r.t. the input x. Such that, J_y{l}(i,j) is the
        %gradient of the j'th y output, of the l'th later, w.r.t. x_i
        %Size m by n, with m being size of y (the layer) and n being dimensionality of problem. 
    %J_y: J_y: matrix, such that J_y(i,j) is partial derivative of y_L(i)
        %w.r.t. x_j

%NOTES
    %the biases are not present, because the partial derivative was taken
    %(they're constants, and disappear
    
    
% We do this as a weird forward pass defined by taking partial derivative
% of the usual feedforward pass w.r.t. x_i and working with that. 
    
    %l for layers
    for l=1
        J_z{l}=weights{1};
    end
    
    for l=2:(length(weights)-1) %mult w/ weight matrix, nonlinearity
        J_z{l}=weights{l}*diag(arrayfun(@(x) sigmoid_prime(x), z{l-1}))*J_z{l-1};
    end
    
    for l=length(weights) %no nonlinearity at last step
        J_z{l}=weights{l}*diag(arrayfun(@(x) sigmoid_prime(x), z{l-1}))*J_z{l-1};
        J_y=diag(arrayfun(@(x) sigmoid_prime(x), z{l-1}))*J_z{l};
    end


end

