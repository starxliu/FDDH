function [ Y ] = e_dragging( X,margin1,margin2 )

%e_dragging operation
Y = X;
Y((X < margin1)) = margin1;
Y((X > margin2)) = margin2; 

end