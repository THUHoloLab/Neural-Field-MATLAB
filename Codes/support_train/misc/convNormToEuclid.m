function t = convNormToEuclid(s)
    %t = s
    t = (s < 0.5) .* (2*s) + (0.5 <= s) .* (1./(2 - 2*s));
end