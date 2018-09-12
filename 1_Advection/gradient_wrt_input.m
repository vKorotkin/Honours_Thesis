function [J_y, J_z] = gradient_wrt_input(weights, x)
%GRADIENT_WRT_INPUT Summary of this function goes here
%   Detailed explanation goes here\
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
    %matrix of y(l) w.r.t. the input x. Size m by n, with m being size of y
    %(the layer) and n being dimensionality of problem. 
    %J_z: same thing but for z
%NOTES
    %y is simply z with nonlinearity applied elementwise. Computed here to
        %avoid dealing with inverses after.
    %the biases are not present, because the partial derivative was taken
    %(they're constants, and disappear
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end

