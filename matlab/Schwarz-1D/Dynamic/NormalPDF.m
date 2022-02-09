function [ y ] = NormalPDF(x, mean, sigma)

y = (sqrt(1/2/pi)/sigma).*exp(-(x-mean).*(x-mean)./(2*sigma*sigma));