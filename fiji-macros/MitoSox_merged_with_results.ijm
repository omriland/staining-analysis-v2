//Created on Sep 2025
//Use this file and not the other macros as this is the final and merged one

// --- Capture original path so we can reopen later ---
dir  = getInfo("image.directory");
orig = getTitle();                 // e.g., "CSwp3iCTRp33_MITOSOX_CTR.tif"
path = dir + orig;

// ===== STEP 1: your recorded segmentation on BLUE =====
run("Split Channels");
selectWindow(orig + " (green)"); close();
selectWindow(orig + " (red)");   close();

selectWindow(orig + " (blue)");
setAutoThreshold("Otsu dark no-reset");
//run("Threshold...");
//setThreshold(33, 255);
setOption("BlackBackground", true);
run("Convert to Mask");
run("Close-"); // morphological Close (not File>Close)

// Set measurements before AP and avoid stale redirects
run("Set Measurements...", "area mean min redirect=None decimal=3");


run("Analyze Particles...", "size=1000-Infinity pixel circularity=0.1-1.00 exclude clear add show=Nothing");


// ===== STEP 2: close images, reopen original, split, measure ROIs =====
run("Close All");                     // ROI Manager stays open
open(path);                           // reopen the original TIFF
run("Split Channels");

// Build channel names
blue  = orig + " (blue)";
red   = orig + " (red)";
green = orig + " (green)";

// Close green (as you did)
selectWindow(green); close();

// Measure ROIs with redirect to BLUE
selectWindow(red); // activate a different window than redirect target
run("Set Measurements...", "area mean min redirect=[" + blue + "] decimal=3");
roiManager("Measure");

// Measure ROIs with redirect to RED
selectWindow(blue);
run("Set Measurements...", "area mean min redirect=[" + red + "] decimal=3");
roiManager("Measure");

// ===== SAVE RESULTS TO CSV IN ORIGINAL FOLDER =====
base = orig;
dot = lastIndexOf(base, ".");
if (dot > -1) base = substring(base, 0, dot); // strip extension
outPath = dir + base + "_results.csv";
saveAs("Results", outPath);
print("Saved results to: " + outPath);