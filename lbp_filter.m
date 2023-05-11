function out = lbp_filter(img)
    img = im2gray(img);
    [rows, columns] = size(img);
    out = zeros(size(img), 'uint8');

    for row = 2 : rows - 1
        for col = 2 : columns - 1
		    centerPixel = img(row, col);
            pixel7=img(row-1, col-1) > centerPixel;
		    pixel6=img(row-1, col) > centerPixel;
		    pixel5=img(row-1, col+1) > centerPixel;
		    pixel3=img(row+1, col+1) > centerPixel;
		    pixel1=img(row+1, col-1) > centerPixel;
		    pixel0=img(row, col-1) > centerPixel;
    
            % Create LBP image with the starting, LSB pixel in the upper middle.
		    eightBitNumber = uint8(...
			    pixel6 * 2^7 + pixel5 * 2^6 + ...
			    pixel5 * 2^4 + pixel3 * 2^4 + ...
			    pixel3 * 2^2 + pixel1 * 2^2 + ...
			    pixel0 * 2 + pixel7);
		    out(row, col) = eightBitNumber;
        end
    end
end