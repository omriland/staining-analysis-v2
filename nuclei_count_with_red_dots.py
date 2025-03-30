    def process_image(self, image, blue_channel):
        """Process image to detect nuclei and red dots"""
        # Apply Gaussian blur to reduce noise
        blurred_blue = cv2.GaussianBlur(blue_channel, (5, 5), 0)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(blurred_blue, self.params['blue_threshold'], 255, cv2.THRESH_BINARY)

        # Apply morphological operations to connect fragments
        if self.params['closing_size'] > 0:
            closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (self.params['closing_size'], self.params['closing_size']))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, closing_kernel)

        if self.params['dilation_size'] > 0:
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (self.params['dilation_size'], self.params['dilation_size']))
            binary = cv2.dilate(binary, dilation_kernel, iterations=1)
            binary = cv2.erode(binary, dilation_kernel, iterations=1)

        # Label connected components
        labels = measure.label(binary)
        regions = measure.regionprops(labels)

        # Filter by size and create a mask for valid nuclei
        valid_labels = [region.label for region in regions
                        if self.params['min_size'] <= region.area <= self.params['max_size']]

        nuclei_mask = np.zeros_like(binary)
        for label in valid_labels:
            nuclei_mask[labels == label] = 255

        # Create a result image with outlines
        if len(image.shape) == 2:  # Grayscale
            result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result_img = image.copy()

        # Find contours of nuclei
        contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and number them
        for i, contour in enumerate(contours):
            cv2.drawContours(result_img, [contour], -1, (0, 255, 255), 2)
            # Draw the nuclei numbers but not size data
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result_img, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Add manually marked nuclei
        for i, (x, y) in enumerate(self.manual_nuclei):
            # Draw a circle for each manually added nucleus
            cv2.circle(result_img, (x, y), 15, (0, 255, 255), 2)
            # Label with number continuing from automatic detections
            cv2.putText(result_img, str(len(contours) + i + 1), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Process red channel if the image is RGB
        red_dots = []
        green_dots = []
        total_green_area_microns = 0  # Initialize total green area
        
        if len(image.shape) == 3 and image.shape[2] >= 3:
            # Extract red channel
            red_channel = image[:, :, 0]  # red channel is index 0 in RGB
            
            # Apply Gaussian blur
            blurred_red = cv2.GaussianBlur(red_channel, (5, 5), 0)
            
            # Apply threshold to get binary image of red dots
            _, red_binary = cv2.threshold(blurred_red, self.params['red_threshold'], 255, cv2.THRESH_BINARY)
            
            # Label connected components
            red_labels = measure.label(red_binary)
            red_regions = measure.regionprops(red_labels)
            
            # Filter red dots by size
            valid_red_labels = [region.label for region in red_regions
                             if self.params['red_min_size'] <= region.area <= self.params['red_max_size']]
            
            # Create a mask for valid red dots
            red_mask = np.zeros_like(red_binary)
            for label in valid_red_labels:
                red_mask[red_labels == label] = 255
            
            # Find contours of red dots
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on result image
            for i, contour in enumerate(red_contours):
                # Calculate size in microns for storage, but don't display it
                area_pixels = cv2.contourArea(contour)
                area_microns = area_pixels / (self.MICRON_CONVERSION * self.MICRON_CONVERSION)
                
                # Store the dot data
                red_dots.append({
                    'id': i + 1,
                    'area_pixels': area_pixels,
                    'area_microns': area_microns,
                    'contour': contour
                })
                
                # Draw yellow contour only - no text labels for size
                cv2.drawContours(result_img, [contour], -1, (0, 255, 255), 2)
            
            # Always extract green channel for total area, regardless of analyze_green_dots setting
            green_channel = image[:, :, 1]  # green channel is index 1 in RGB
            
            # Default threshold for green channel if not analyzing green dots
            green_threshold = self.params.get('green_threshold', 100)
            
            # Apply Gaussian blur
            blurred_green = cv2.GaussianBlur(green_channel, (5, 5), 0)
            
            # Apply threshold to get binary image of green areas
            _, green_binary = cv2.threshold(blurred_green, green_threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate total green area in microns (all pixels above threshold)
            total_green_pixels = np.sum(green_binary > 0)
            total_green_area_microns = total_green_pixels / (self.MICRON_CONVERSION * self.MICRON_CONVERSION)
            
            # Process green dots if requested (individual dots analysis)
            if self.analyze_green_dots:
                # Label connected components for individual dots
                green_labels = measure.label(green_binary)
                green_regions = measure.regionprops(green_labels)
                
                # Filter green dots by size
                valid_green_labels = [region.label for region in green_regions
                                 if self.params['green_min_size'] <= region.area <= self.params['green_max_size']]
                
                # Create a mask for valid green dots
                green_mask = np.zeros_like(green_binary)
                for label in valid_green_labels:
                    green_mask[green_labels == label] = 255
                
                # Find contours of green dots
                green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours on result image
                for i, contour in enumerate(green_contours):
                    # Calculate size in microns for storage, but don't display it
                    area_pixels = cv2.contourArea(contour)
                    area_microns = area_pixels / (self.MICRON_CONVERSION * self.MICRON_CONVERSION)
                    
                    # Store the dot data
                    green_dots.append({
                        'id': i + 1,
                        'area_pixels': area_pixels,
                        'area_microns': area_microns,
                        'contour': contour
                    })
                    
                    # Draw magenta contour for green dots (to distinguish from red) - no text labels
                    cv2.drawContours(result_img, [contour], -1, (255, 0, 255), 2)
        
        # Save detected dots
        self.red_dots = red_dots
        if self.analyze_green_dots:
            self.green_dots = green_dots
            
        # Store total green area regardless of analyze_green_dots setting
        self.total_green_area_microns = total_green_area_microns

        return binary, nuclei_mask, result_img, len(contours) + len(self.manual_nuclei) 