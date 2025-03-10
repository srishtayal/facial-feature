imgFile = 'photo.jpg'; 
if exist(imgFile, 'file') ~= 2
    error(['Image file "', imgFile, '" not found.']);
end

img = imread(imgFile);
figure, imshow(img);
title('Original Image');

faceDetector = vision.CascadeObjectDetector(); 
bboxFace = step(faceDetector, img);

if isempty(bboxFace)
    error('No face detected in the image.');
end

imgFace = insertObjectAnnotation(img, 'rectangle', bboxFace, 'Face');
figure, imshow(imgFace);
title('Detected Face');

faceX = bboxFace(1); faceY = bboxFace(2);
faceW = bboxFace(3); faceH = bboxFace(4);

eyeDetector = vision.CascadeObjectDetector('EyePairBig');
eyeBbox = step(eyeDetector, img);

if ~isempty(eyeBbox)
    imgEye = insertObjectAnnotation(img, 'rectangle', eyeBbox(1, :), 'Eyes');
    figure, imshow(imgEye);
    title('Detected Eyes');
end

noseDetector = vision.CascadeObjectDetector('Nose');
noseBboxes = step(noseDetector, img);

if ~isempty(noseBboxes)
    noseBbox = chooseClosestFeature(noseBboxes, bboxFace);
    imgNose = insertObjectAnnotation(img, 'rectangle', noseBbox, 'Nose');
    figure, imshow(imgNose);
    title('Detected Nose');
end

mouthDetector = vision.CascadeObjectDetector('Mouth');
mouthBboxes = step(mouthDetector, img);

if ~isempty(mouthBboxes)
    mouthBbox = chooseClosestFeature(mouthBboxes, bboxFace, 'below');
    imgMouth = insertObjectAnnotation(img, 'rectangle', mouthBbox, 'Mouth');
    figure, imshow(imgMouth);
    title('Detected Mouth');
end

disp('Facial Feature Recognition Completed.');

function bestBbox = chooseClosestFeature(bboxes, faceBbox, position)
    if nargin < 3
        position = 'center';
    end

    faceX = faceBbox(1); faceY = faceBbox(2);
    faceW = faceBbox(3); faceH = faceBbox(4);
    faceCenter = [faceX + faceW/2, faceY + faceH/2];

    minDist = inf;
    bestBbox = bboxes(1, :);

    for i = 1:size(bboxes, 1)
        featureX = bboxes(i, 1) + bboxes(i, 3) / 2;
        featureY = bboxes(i, 2) + bboxes(i, 4) / 2;
        
        if strcmp(position, 'below') && featureY < (faceY + faceH * 0.7)
            continue;
        end

        dist = sqrt((featureX - faceCenter(1))^2 + (featureY - faceCenter(2))^2);
        
        if dist < minDist
            minDist = dist;
            bestBbox = bboxes(i, :);
        end
    end
end
