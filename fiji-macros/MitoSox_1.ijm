orig = getTitle();
run("Split Channels");

selectWindow(orig + " (green)"); close();
selectWindow(orig + " (red)");   close();

selectWindow(orig + " (blue)");
setAutoThreshold("Otsu dark no-reset");
//run("Threshold...");
//setThreshold(33, 255);
setOption("BlackBackground", true);
run("Convert to Mask");
run("Close-"); // <-- binary morphological Close (not File>Close)

// set measurements before AP; redirect=None avoids the old redirect error
run("Set Measurements...", "area mean min redirect=None decimal=3");

// keep your two AP calls
run("Analyze Particles...", "size=1000-infinity pixel circularity=0.1-1.00 display exclude clear summarize add");
