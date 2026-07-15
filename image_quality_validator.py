"""
================================================================================
IMAGE QUALITY VALIDATOR - RETINAGUARD V500
================================================================================
Prevents "Garbage In, Garbage Out" by validating fundus image quality BEFORE
clinical feature extraction begins.

VALIDATION CHECKS:
  1. Blur Detection (Laplacian variance)
  2. Brightness/Contrast (mean intensity, dynamic range)
  3. Resolution (minimum pixel dimensions)
  4. Color Distribution (RGB channel balance)
  5. Vignetting (peripheral darkness check)
  6. Vessel Network Connectivity (ensures optic disc visible)

REJECTION CRITERIA:
  - Blurry images (variance of Laplacian < 100)
  - Over/underexposed (mean < 30 or > 220)
  - Low resolution (< 512×512 pixels)
  - Excessive vignetting (peripheral mean < 30% of center)
  - Missing vessel network (no optic disc detected)

Version: 1.0.0
Author: RetinaGuard Development Team
================================================================================
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import sys

class ImageQualityValidator:
    """Validate fundus image quality before clinical analysis"""
    
    # Quality thresholds (configurable)
    BLUR_THRESHOLD = 100.0          # Laplacian variance (lower = blurrier)
    MIN_BRIGHTNESS = 30             # Minimum mean intensity (0-255)
    MAX_BRIGHTNESS = 220            # Maximum mean intensity (0-255)
    MIN_RESOLUTION = 512            # Minimum width/height in pixels
    OPTIMAL_RESOLUTION = 1024       # Recommended resolution
    MIN_DYNAMIC_RANGE = 25          # Minimum std deviation (contrast)
    MAX_VIGNETTING_RATIO = 0.30     # Max ratio of peripheral/center darkness
    MIN_VESSEL_DENSITY = 0.05       # Minimum vessel network coverage
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator
        
        Args:
            strict_mode: If True, reject borderline quality images
                        If False, allow borderline images with warnings
        """
        self.strict_mode = strict_mode
        
        # If strict_mode is False (e.g., handheld camera or smartphone),
        # relax the physical hardware thresholds because these cameras
        # inherently produce darker, blurrier, and flatter images.
        if not strict_mode:
            self.BLUR_THRESHOLD = 5.0          # Allow massive blur
            self.MIN_BRIGHTNESS = 5            # Allow massive underexposure
            self.MIN_DYNAMIC_RANGE = 5         # Allow flat contrast
            self.MIN_VESSEL_DENSITY = 0.001    # Allow poor vessel visibility
            self.MAX_VIGNETTING_RATIO = 0.50   # Allow heavier vignetting
            
    def validate(self, image: np.ndarray, patient_id: str = "UNKNOWN") -> Dict:
        """
        Comprehensive image quality validation
        
        Args:
            image: Input fundus image (BGR or RGB format)
            patient_id: Patient identifier for logging
            
        Returns:
            Dictionary with:
                - 'valid': bool (True if image passes all checks)
                - 'quality_score': float (0-100, overall quality rating)
                - 'warnings': list of warning messages
                - 'errors': list of error messages (reasons for rejection)
                - 'metrics': dict of measured quality metrics
        """
        warnings = []
        errors = []
        metrics = {}
        
        print(f"\n   [Q] IMAGE QUALITY VALIDATION for {patient_id}", flush=True)
        print(f"      {'='*60}", flush=True)
        
        # Convert to grayscale for some checks
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # CHECK 0: Structural Integrity (Security/OOD Check)
        # Rejects non-eye images (e.g., selfies, dogs) by checking for the circular FOV mask.
        h, w = gray.shape
        corner_size = max(10, min(h, w) // 10)
        
        corners = [
            gray[0:corner_size, 0:corner_size],
            gray[0:corner_size, w-corner_size:w],
            gray[h-corner_size:h, 0:corner_size],
            gray[h-corner_size:h, w-corner_size:w]
        ]
        
        # Fundus images are circles in a black square (corners are pitch black). Natural images have bright corners.
        corner_brightness = np.mean([np.mean(c) for c in corners])
        metrics['corner_brightness'] = round(corner_brightness, 2)
        
        print(f"      [0] Structural Security: corner_brightness={corner_brightness:.1f}", end=" -> ", flush=True)
        
        # If strict mode is off, we are more lenient for cropped images
        security_threshold = 85.0 if self.strict_mode else 120.0
        
        if corner_brightness > security_threshold:
            print(f"[X] CRITICAL SECURITY FAILURE", flush=True)
            print(f"\\n      [X] VERDICT: REJECTED - NON-RETINAL IMAGE DETECTED", flush=True)
            print(f"      Reason: The image lacks the characteristic circular Field-Of-View mask of a fundus scan.", flush=True)
            print(f"      {'='*60}\\n", flush=True)
            sys.stdout.flush()
            
            return {
                'valid': False,
                'quality_score': 0.0,
                'warnings': [],
                'errors': [f"CRITICAL SECURITY REJECTION: Image appears to be a natural photo (corner brightness {corner_brightness:.1f} > {security_threshold}), not a fundus scan. Please upload a valid retina image."],
                'metrics': metrics,
                'critical_failure': True,
                'failure_reason': 'OOD_SECURITY_REJECTION'
            }
        else:
            print(f"[+] PASS (Valid Fundus Structure)", flush=True)
            
        # CHECK 1: Resolution
        height, width = gray.shape
        metrics['resolution'] = f"{width}×{height}"
        metrics['width'] = width
        metrics['height'] = height
        
        print(f"      [1] Resolution: {width}x{height} px", end=" -> ", flush=True)
        
        # CRITICAL: Hard resolution requirement (FDA specification compliance)
        # RetinaGuard requires 512×512 minimum for vessel dimensional analysis
        if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
            print(f"[X] CRITICAL FAILURE", flush=True)
            print(f"\n      [X] VERDICT: REJECTED - RESOLUTION BELOW CLINICAL MINIMUM", flush=True)
            print(f"      Required: {self.MIN_RESOLUTION}×{self.MIN_RESOLUTION} | Received: {width}×{height}", flush=True)
            print(f"      Reason: Blood vessel diameter measurement requires minimum pixel density.", flush=True)
            print(f"      {'='*60}\n", flush=True)
            sys.stdout.flush()
            
            # Immediate rejection - do not proceed with further checks
            return {
                'valid': False,
                'quality_score': 0.0,
                'warnings': [],
                'errors': [f"CRITICAL: Resolution {width}×{height} below clinical minimum of {self.MIN_RESOLUTION}×{self.MIN_RESOLUTION}. Vessel measurements require higher pixel density."],
                'metrics': metrics,
                'critical_failure': True,
                'failure_reason': 'INSUFFICIENT_RESOLUTION'
            }
        
        if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
            errors.append(f"Resolution too low ({width}×{height}). Minimum: {self.MIN_RESOLUTION}×{self.MIN_RESOLUTION}")
            print(f"[X] FAIL (too small)", flush=True)
        elif width < self.OPTIMAL_RESOLUTION or height < self.OPTIMAL_RESOLUTION:
            warnings.append(f"Resolution below optimal ({width}x{height}). Recommended: {self.OPTIMAL_RESOLUTION}x{self.OPTIMAL_RESOLUTION}")
            print(f"[!] WARN (below optimal)", flush=True)
        else:
            print(f"[+] PASS", flush=True)
        
        # CHECK 2: Blur Detection (Laplacian Variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_variance = laplacian.var()
        metrics['blur_variance'] = round(blur_variance, 2)
        
        print(f"      [2] Blur Detection: variance={blur_variance:.2f}", end=" -> ", flush=True)
        
        if blur_variance < self.BLUR_THRESHOLD:
            errors.append(f"Image too blurry (variance={blur_variance:.2f}). Minimum: {self.BLUR_THRESHOLD}")
            print(f"[X] FAIL (out of focus)", flush=True)
        elif blur_variance < self.BLUR_THRESHOLD * 1.5:
            warnings.append(f"Image slightly blurry (variance={blur_variance:.2f})")
            print(f"[!] WARN (borderline focus)", flush=True)
        else:
            print(f"[+] PASS (sharp)", flush=True)
        
        # CHECK 3: Brightness/Exposure
        mean_brightness = gray.mean()
        metrics['mean_brightness'] = round(mean_brightness, 2)
        
        print(f"      [3] Brightness: mean={mean_brightness:.1f}", end=" -> ", flush=True)
        
        if mean_brightness < self.MIN_BRIGHTNESS:
            errors.append(f"Image too dark (mean={mean_brightness:.1f}). Minimum: {self.MIN_BRIGHTNESS}")
            print(f"[X] FAIL (underexposed)", flush=True)
        elif mean_brightness > self.MAX_BRIGHTNESS:
            errors.append(f"Image too bright (mean={mean_brightness:.1f}). Maximum: {self.MAX_BRIGHTNESS}")
            print(f"[X] FAIL (overexposed)", flush=True)
        elif mean_brightness < self.MIN_BRIGHTNESS + 20 or mean_brightness > self.MAX_BRIGHTNESS - 20:
            warnings.append(f"Brightness borderline (mean={mean_brightness:.1f}). Optimal: {self.MIN_BRIGHTNESS+20}-{self.MAX_BRIGHTNESS-20}")
            print(f"[!] WARN (exposure borderline)", flush=True)
        else:
            print(f"[+] PASS", flush=True)
        
        # CHECK 4: Contrast/Dynamic Range
        std_brightness = gray.std()
        metrics['std_brightness'] = round(std_brightness, 2)
        
        print(f"      [4] Contrast: std={std_brightness:.1f}", end=" -> ", flush=True)
        
        if std_brightness < self.MIN_DYNAMIC_RANGE:
            errors.append(f"Insufficient contrast (std={std_brightness:.1f}). Minimum: {self.MIN_DYNAMIC_RANGE}")
            print(f"[X] FAIL (flat contrast)", flush=True)
        elif std_brightness < self.MIN_DYNAMIC_RANGE * 1.2:
            warnings.append(f"Low contrast (std={std_brightness:.1f})")
            print(f"[!] WARN (low contrast)", flush=True)
        else:
            print(f"[+] PASS", flush=True)
        
        # CHECK 5: Vignetting (Peripheral Darkness)
        vignetting_ratio = self._check_vignetting(gray)
        metrics['vignetting_ratio'] = round(vignetting_ratio, 3)
        
        print(f"      [5] Vignetting: ratio={vignetting_ratio:.3f}", end=" -> ", flush=True)
        
        if vignetting_ratio < self.MAX_VIGNETTING_RATIO:
            errors.append(f"Excessive vignetting (ratio={vignetting_ratio:.3f}). Maximum: {self.MAX_VIGNETTING_RATIO}")
            print(f"[X] FAIL (dark edges)", flush=True)
        elif vignetting_ratio < self.MAX_VIGNETTING_RATIO + 0.1:
            warnings.append(f"Noticeable vignetting (ratio={vignetting_ratio:.3f})")
            print(f"[!] WARN (slight vignetting)", flush=True)
        else:
            print(f"[+] PASS", flush=True)
        
        # CHECK 6: Color Balance (RGB channels)
        if len(image.shape) == 3:
            color_balance = self._check_color_balance(image)
            metrics['color_balance'] = color_balance
            
            print(f"      [6] Color Balance: R={color_balance['r']:.1f} G={color_balance['g']:.1f} B={color_balance['b']:.1f}", end=" -> ", flush=True)
            
            # Check for severe color casts
            max_diff = max(abs(color_balance['r'] - color_balance['g']),
                          abs(color_balance['g'] - color_balance['b']),
                          abs(color_balance['b'] - color_balance['r']))
            
            if max_diff > 50:
                warnings.append(f"Color cast detected (max channel diff={max_diff:.1f})")
                print(f"[!] WARN (color cast)", flush=True)
            else:
                print(f"[+] PASS", flush=True)
        
        # CHECK 7: Vessel Network Detection (Ensures vascularization)
        vessel_coverage = self._estimate_vessel_coverage(gray)
        metrics['vessel_coverage'] = round(vessel_coverage, 4)
        
        print(f"      [7] Vessel Network: coverage={vessel_coverage:.4f}", end=" -> ", flush=True)
        
        if vessel_coverage < self.MIN_VESSEL_DENSITY:
            print(f"[X] CRITICAL SECURITY FAILURE", flush=True)
            print(f"\\n      [X] VERDICT: REJECTED - NO BLOOD VESSELS DETECTED", flush=True)
            print(f"      Reason: The image lacks a retinal blood vessel network (Coverage: {vessel_coverage:.4f}). This is likely a non-eye object.", flush=True)
            print(f"      {'='*60}\\n", flush=True)
            sys.stdout.flush()
            
            return {
                'valid': False,
                'quality_score': 0.0,
                'warnings': [],
                'errors': [f"CRITICAL SECURITY REJECTION: No retinal blood vessels detected. This appears to be a non-eye object."],
                'metrics': metrics,
                'critical_failure': True,
                'failure_reason': 'OOD_NO_VESSELS'
            }
        elif vessel_coverage < self.MIN_VESSEL_DENSITY * 1.5:
            warnings.append(f"Weak vessel network (coverage={vessel_coverage:.4f})")
            print(f"[!] WARN (weak vessels)", flush=True)
        else:
            print(f"[+] PASS", flush=True)
            
        # CHECK 8: Anatomical Security (Optic Disc Detection)
        # Prevents adversarial attacks (like a fertilized chicken egg) which have vessels but no optic disc.
        has_optic_disc = self._detect_optic_disc(gray)
        metrics['has_optic_disc'] = has_optic_disc
        
        print(f"      [8] Anatomical Security: Optic Disc=", end="", flush=True)
        if not has_optic_disc:
            print(f"MISSING -> [X] CRITICAL SECURITY FAILURE", flush=True)
            print(f"\\n      [X] VERDICT: REJECTED - NO OPTIC DISC DETECTED", flush=True)
            print(f"      Reason: The image has vessels but lacks a human optic disc (e.g., adversarial chicken egg attack).", flush=True)
            print(f"      {'='*60}\\n", flush=True)
            sys.stdout.flush()
            
            return {
                'valid': False,
                'quality_score': 0.0,
                'warnings': [],
                'errors': ["CRITICAL SECURITY REJECTION: No Optic Disc detected. Adversarial non-human object suspected."],
                'metrics': metrics,
                'critical_failure': True,
                'failure_reason': 'OOD_NO_OPTIC_DISC'
            }
        else:
            print(f"DETECTED -> [+] PASS", flush=True)
        
        # CALCULATE OVERALL QUALITY SCORE (0-100)
        quality_score = self._calculate_quality_score(metrics, errors, warnings)
        
        # DETERMINE VALIDITY
        valid = len(errors) == 0
        if not valid:
            print(f"\n      [X] VERDICT: REJECTED (Quality Score: {quality_score:.1f}/100)", flush=True)
            print(f"      Errors: {len(errors)} | Warnings: {len(warnings)}", flush=True)
        elif len(warnings) > 0:
            print(f"\n      [!] VERDICT: ACCEPTED WITH WARNINGS (Quality Score: {quality_score:.1f}/100)", flush=True)
            print(f"      Warnings: {len(warnings)}", flush=True)
        else:
            print(f"\n      [+] VERDICT: EXCELLENT QUALITY (Quality Score: {quality_score:.1f}/100)", flush=True)
        
        print(f"      {'='*60}\n", flush=True)
        sys.stdout.flush()
        
        return {
            'valid': valid,
            'quality_score': quality_score,
            'warnings': warnings,
            'errors': errors,
            'metrics': metrics
        }
    
    def _check_vignetting(self, gray: np.ndarray) -> float:
        """
        Measure vignetting (peripheral darkening)
        
        Returns:
            Ratio of peripheral brightness to center brightness (0-1)
            Lower values = more vignetting
        """
        h, w = gray.shape
        
        # Define center region (middle 50%)
        center_y1, center_y2 = int(h * 0.25), int(h * 0.75)
        center_x1, center_x2 = int(w * 0.25), int(w * 0.75)
        center_region = gray[center_y1:center_y2, center_x1:center_x2]
        center_mean = center_region.mean()
        
        # Define peripheral region (outer 15% border)
        border_width = int(min(h, w) * 0.15)
        top_border = gray[:border_width, :]
        bottom_border = gray[-border_width:, :]
        left_border = gray[:, :border_width]
        right_border = gray[:, -border_width:]
        
        peripheral_mean = np.mean([
            top_border.mean(),
            bottom_border.mean(),
            left_border.mean(),
            right_border.mean()
        ])
        
        # Ratio (0 = complete darkness at edges, 1 = uniform brightness)
        if center_mean > 0:
            return peripheral_mean / center_mean
        else:
            return 0.0
    
    def _check_color_balance(self, image: np.ndarray) -> Dict[str, float]:
        """Check RGB channel balance"""
        b, g, r = cv2.split(image)
        return {
            'r': r.mean(),
            'g': g.mean(),
            'b': b.mean()
        }
    
    def _estimate_vessel_coverage(self, gray: np.ndarray) -> float:
        """
        Estimate vessel network coverage (simplified)
        
        Uses edge detection to approximate vessel presence
        """
        # Apply CLAHE to enhance vessels
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection (vessels appear as edges)
        edges = cv2.Canny(enhanced, 30, 100)
        
        # Calculate coverage (% of pixels that are vessel edges)
        coverage = np.sum(edges > 0) / edges.size
        
        return coverage
    
    def _detect_optic_disc(self, gray: np.ndarray) -> bool:
        """
        Detects the presence of an Optic Disc (the brightest contiguous region in a retina).
        This defeats adversarial attacks (like an egg yolk) which have no optic disc.
        """
        # Apply slight blur to remove noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Find the brightest 1% of pixels (potential optic disc)
        max_val = np.max(blurred)
        _, bright_mask = cv2.threshold(blurred, max_val - 30, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
            
        # The optic disc should be a reasonably sized, somewhat circular blob
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Must be between 50 and 50000 pixels (to ignore small noise and giant flashes)
            if 50 < area < 50000:
                # Check circularity (Optic Disc is roughly circular/oval)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.3:  # 1.0 is a perfect circle. 0.3 allows for ovals.
                        return True
                        
        return False
    
    def _calculate_quality_score(self, metrics: Dict, errors: list, warnings: list) -> float:
        """
        Calculate overall quality score (0-100)
        
        Scoring:
            - Start at 100
            - Deduct 20 points per error
            - Deduct 5 points per warning
            - Bonus points for high resolution, sharpness
        """
        score = 100.0
        
        # Penalties
        score -= len(errors) * 20
        score -= len(warnings) * 5
        
        # Bonuses for exceptional quality
        if 'blur_variance' in metrics and metrics['blur_variance'] > 500:
            score += 5  # Very sharp image
        
        if 'width' in metrics and metrics['width'] >= 2048:
            score += 5  # High resolution
        
        # Clamp to 0-100
        return max(0.0, min(100.0, score))


def validate_image_quality(image: np.ndarray, patient_id: str = "UNKNOWN", strict: bool = True) -> Dict:
    """
    Convenience function for external use
    
    Args:
        image: Input fundus image (BGR or RGB format)
        patient_id: Patient identifier for logging
        strict: Strict validation mode
        
    Returns:
        Validation result dictionary
    """
    validator = ImageQualityValidator(strict_mode=strict)
    return validator.validate(image, patient_id)


# Testing harness
if __name__ == "__main__":
    print("="*80, flush=True)
    print("IMAGE QUALITY VALIDATOR - TEST SUITE", flush=True)
    print("="*80, flush=True)
    
    # Test Case 1: Normal quality image
    test_image = np.random.randint(50, 200, size=(1024, 1024, 3), dtype=np.uint8)
    result = validate_image_quality(test_image, "TEST-001")
    print(f"\nTest 1 - Normal Image: {'PASS' if result['valid'] else 'FAIL'}", flush=True)
    
    # Test Case 2: Blurry image (apply heavy Gaussian blur)
    blurry = cv2.GaussianBlur(test_image, (51, 51), 0)
    result = validate_image_quality(blurry, "TEST-002")
    print(f"Test 2 - Blurry Image: {'REJECTED (expected)' if not result['valid'] else 'UNEXPECTED PASS'}", flush=True)
    
    # Test Case 3: Low resolution
    low_res = cv2.resize(test_image, (256, 256))
    result = validate_image_quality(low_res, "TEST-003")
    print(f"Test 3 - Low Resolution: {'REJECTED (expected)' if not result['valid'] else 'UNEXPECTED PASS'}", flush=True)
    
    # Test Case 4: Overexposed
    overexposed = np.ones((1024, 1024, 3), dtype=np.uint8) * 240
    result = validate_image_quality(overexposed, "TEST-004")
    print(f"Test 4 - Overexposed: {'REJECTED (expected)' if not result['valid'] else 'UNEXPECTED PASS'}", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("VALIDATION COMPLETE", flush=True)
    print("="*80, flush=True)
