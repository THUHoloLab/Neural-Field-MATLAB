function s = convEuclidToNorm(t)
    %t = s;
    s = (t < 1) .* (t/2) +  (1 <= t) .* (1 - 1./(2*t));
end