function [y, z, output]=forward_pass(x, weights, biases)
%FORWARD PASS OF NEURAL NETWORK
    %Go from start to end of neural network, computing the output. 
    %We also need the intermediate quantities i.e. inputs and outputs of
    %the neurons, as well as their derivatives (Jacobians) w.r.t inputs
%DEFINITIONS: 
    %L: number of layers of neural network. 
%INPUTS
    %x: M by 1 vector, input to neural net
    %weights and biases characterize the neural network. 
        %weights: cell array, size 1 to L where L is number of layers. The l'th
            %entry contains matrix W(l) which connects the l'th layer to the l-1th
        %biases: cell array, size 1 to L where L is number of layers. The l'th
            %entry contains biases b(l), such that output is sigmoid(W y(l-1)+b)
            %where y(l-1) is output of previous layer. 
%OUTPUTS
    %y: cell array with size L. y{l} is a vector, output of layer l. 
    %z: cell array with size L. z{l} is a vector, weighted input to layer l.
    %output: vector, y{L}, overall output of network. 
%NOTES
    %y is simply z with nonlinearity applied elementwise. Computed here to
        %avoid dealing with inverses after. 

    y=biases; z=biases; %need shape for arrays
    
    %y: activations for each neuron. last cell array element empty, no
    %nonlinearity at output
    %z: weighted inputs for each neuron. first cell array element empty, no
    %"weighted input" at input
    
    %l for layers
    for l=1
        y{l}=x; %have an output but no weighted input
        z{l}=x;
    end
    
    for l=2:(length(weights)-1) %mult w/ weight matrix, nonlinearity
        z{l}=weights{l}*y{l-1}+biases{l};
        y{l}=arrayfun(@(inp) sigmoid(inp), z{l});
    end
    
    for l=length(weights) %no nonlinearity at last step
        z{l}=weights{l}*y{l-1}+biases{l};
        y{l}=z{l};
        output=y{l};
    end
    
end

