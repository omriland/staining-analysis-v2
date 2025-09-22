// Assumes: ROI Manager already contains ROIs.
// Start with your original multichannel image open.
orig = getTitle();           // e.g., "CSwp3iCTRp33_MITOSOX_CTR.tif"
run("Split Channels");

// Build channel window names
blue  = orig + " (blue)";
red   = orig + " (red)";
green = orig + " (green)";

// Close green (as in your recording)
selectWindow(green); close();

// 1) Measure with redirect to BLUE
//    (activate RED so redirect points to a different window)
selectWindow(red);
run("Set Measurements...", "area mean min redirect=[" + blue + "] decimal=3");
roiManager("Measure");

// 2) Measure with redirect to RED
//    (activate BLUE so redirect points to a different window)
selectWindow(blue);
run("Set Measurements...", "area mean min redirect=[" + red + "] decimal=3");
roiManager("Measure");