# ADS-B Aircraft Conflict Prediction System

## 1. Project Overview

* Project Title: ADS-B Aircraft Near-Miss and Conflict Prediction System
* Objective: Predict mid-air loss of separation between aircraft in airport terminal airspace under real-world sensor inaccuracies.
* Core Innovation: Combines spatial indexing, physical sensor uncertainty modeling, Monte Carlo simulation, and a PyTorch Long Short-Term Memory (LSTM) sequence neural network.

---

## 2. Dataset and Ground Truth Labeling

### Does the raw dataset contain conflict labels?
No. The raw TrajAir dataset contains only 7 columns: Frame, ID, x (in km), y (in km), z (in km), Headwind, and Crosswind.

### How are the ground truth labels created?
Labels are generated programmatically using official FAA and ICAO terminal airspace separation rules via the Closest Point of Approach (CPA) algorithm:
* Horizontal separation minimum: 5.0 nautical miles (approx 9.26 km)
* Vertical separation minimum: 1,000 feet
* If two aircraft violate BOTH horizontal and vertical separation limits, the label is CONFLICT (1).
* If either separation limit is safely maintained, the label is SAFE (0).

### Data split and evaluation integrity
* Training Data: Exclusively from the train folder (141 flight scenarios, 51 multi-aircraft files, generating 15,739 trajectory sequences).
* Testing Data: Exclusively from the test folder (62 flight scenarios, 26 multi-aircraft files, generating 4,295 trajectory sequences).
* Strict file-level separation: Training and test scenarios are completely separated by file. The model never sees test flights during training, ensuring zero data leakage.

### What does "Files with 2+ aircraft: 26" mean?
A collision can only occur between two or more planes. Many raw recording files contain only one solo aircraft in the airspace. Out of 62 files in the test directory, exactly 26 files contain 2 or more aircraft simultaneously.

### Are test metrics calculated on one file or all files?
All test metrics are calculated across all 26 test files combined (4,295 total sequences), not on just one single file.

### What does "Majority-Class Baseline: 66.80%" mean?
In the 4,295 test sequences, 2,869 sequences are conflicts and 1,426 are safe.
The conflict ratio is 2,869 / 4,295 = 66.80%.
A naive model that always guesses CONFLICT without looking at radar or altitude achieves 66.80% accuracy.
Our LSTM model achieves 98.30% accuracy, proving an improvement of +31.50% over the baseline.

---

## 3. Why LSTM Replaced MLP

In the initial phase, a Multi-Layer Perceptron (MLP) was used. It was upgraded to an LSTM for critical operational reasons:

1. Static Snapshot vs Temporal History:
   * MLP looks at only 1 single static frame (instant photograph). It cannot tell if an aircraft is accelerating, decelerating, or turning.
   * LSTM processes 10 consecutive seconds of flight history (10 frames x 14 features). It tracks turning rates, closing speed, and climb/descent profiles.
2. Aviation Feature Set (14 Features per Timestep):
   * Aircraft A state: x position, y position, vx velocity, vy velocity, altitude
   * Aircraft B state: x position, y position, vx velocity, vy velocity, altitude
   * Relative dynamics: horizontal distance, closing speed, vertical altitude gap, relative heading angle
3. Training Scale:
   * MLP was trained on 4,187 single frames.
   * LSTM is trained on 15,739 10-second sequences.
4. Model Performance on Unseen Test Flights:
   * Majority Baseline: 66.80%
   * LSTM Accuracy: 98.30%
   * LSTM Precision: 98.44%
   * LSTM Recall: 99.02% (correctly catches 2,841 out of 2,869 conflict sequences)

---

## 4. The 5-Stage Operational Pipeline

### Stage 1: H3 Hexagonal Spatial Indexing (h3_spatial.py)
* Problem: Checking all aircraft pairs across the sky requires O(N^2) calculations, which slows down significantly as aircraft count grows.
* Solution: Uber H3 discrete global hexagonal grid system.
* What is Resolution 6?
  * Each hexagonal cell has a diameter of approx 7.4 km (area approx 36.4 sq km).
  * 1-ring neighborhood (the cell plus its 6 immediate neighbors) spans approx 15 to 20 km.
  * This span safely encloses the FAA 5.0 nautical mile (9.26 km) conflict buffer.
* Pruning: Only aircraft in the same or neighboring hexagons are evaluated. Aircraft in distant hexagons are pruned immediately, cutting pair checks by 60% to 90%.

### Stage 2: Sensor Uncertainty Modeling (uncertainty.py)
* Problem: ADS-B GPS and barometric sensors are not perfectly exact in the real world.
* Standard: Configured using FAA / RTCA DO-260B specifications:
  * Horizontal GPS noise: Gaussian standard deviation of 100 meters (0.1 km)
  * Vertical altitude noise: Gaussian standard deviation of 100 feet
  * Velocity vector noise: Gaussian standard deviation of 5 km/h

### Stage 3: Monte Carlo Simulation (monte_carlo.py)
* Problem: A single deterministic formula gives a rigid binary answer (Yes or No). In borderline cases near 1,000 feet, sensor noise can easily flip the result.
* Solution: Runs 100 to 500 stochastic simulations per aircraft pair. Each run adds random Gaussian sensor noise and checks for conflict.
* Output: The percentage of runs that result in conflict is the Monte Carlo probability (example: 63 out of 100 runs = 63.0%).

### Stage 4: LSTM Sequence Model (conflict_probability.py)
* Runs the trained PyTorch neural network on the 10-second temporal trajectory.
* Outputs the AI probability score based on flight dynamics and closing rates.

### Stage 5: Decision Fusion and Risk Tiers
* Final Probability Formula:
  Final Risk = (0.4 * LSTM Probability) + (0.6 * Monte Carlo Probability)
* Risk Tiers:
  * HIGH RISK: 70% to 100% (immediate collision alert)
  * MEDIUM RISK: 30% to 69% (traffic advisory, converging)
  * LOW RISK: 0% to 29% (safe separation maintained)

---

## 5. Terminal Outputs Explained

### Case A: Output from "python train_conflict_predictor.py"
```text
Sample sequence predictions:
  [OK] Actual: SAFE, Predicted: SAFE (0%)
```
* Actual: SAFE means the two aircraft maintained safe separation.
* Predicted: SAFE means the model correctly predicted safe separation.
* (0%) is the conflict probability. 0% conflict probability means the model is 100% sure the aircraft are SAFE.
* [OK] means the model prediction matched ground truth.

For a conflict scenario, the output looks like:
```text
  [OK] Actual: CONFLICT, Predicted: CONFLICT (99%)
```
* This means the aircraft breached separation, and the LSTM output a 99% probability of conflict.

### Case B: Output from "python conflict_probability.py"
* Case 1 (Head-on, same altitude): Final probability 100.0%, Risk HIGH. CPA, LSTM, and Monte Carlo all agree at 100%.
* Case 2 (Altitude separated by 2,000 ft): Final probability 0.0%, Risk LOW. CPA, LSTM, and Monte Carlo all agree at 0%.
* Case 3 (Borderline separation at 950 ft):
  * CPA says CONFLICT (because 950 ft is less than 1,000 ft).
  * LSTM says 100.0%.
  * Monte Carlo says 63.0% (due to +/- 100 ft sensor noise, 63 runs breached separation and 37 runs stayed clear).
  * Final Fused Probability: (0.4 * 100%) + (0.6 * 63.0%) = 77.8% (HIGH Risk).

---

## 6. Slide-by-Slide PPT Content and Presentation Notes

### Slide 1: Title and Objective
* Slide Title: ADS-B Aircraft Conflict Prediction System
* Bullet Points:
  * Automated near-miss detection in airport terminal airspace.
  * Solves real-world sensor inaccuracies and trajectory dynamics.
  * Architecture combines H3 spatial indexing, Monte Carlo uncertainty, and PyTorch LSTM AI.
  * Evaluated on Pittsburgh-Butler Regional Airport TrajAir dataset.
* Speaker Note:
  "Our project predicts aircraft conflicts in terminal airspace. Instead of assuming perfect GPS or running slow pairwise checks, we use a 5-stage pipeline combining spatial indexing, physical sensor noise modeling, and deep learning."

### Slide 2: Dataset, Labeling, and Baseline
* Slide Title: Data Pipeline and Ground Truth Labeling
* Bullet Points:
  * TrajAir ADS-B data: 203 total flight files at Pittsburgh-Butler Airport.
  * Ground truth labeling via FAA separation standards: 5.0 nm horizontal and 1,000 ft vertical minimums.
  * Strict folder split: 141 train files (15,739 sequences) and 62 test files (4,295 sequences across 26 multi-aircraft files).
  * Zero data leakage: Evaluation is performed strictly on unseen test flight dates.
  * Baseline: A blind majority guess achieves 66.80% accuracy on the test set.
* Speaker Note:
  "Raw ADS-B data does not have collision labels, so we labeled it using official FAA separation standards. All metrics are evaluated across 26 unseen test flight files totaling 4,295 sequences, compared against a 66.80% majority baseline."

### Slide 3: Model Evolution: MLP to LSTM
* Slide Title: Upgrading from Static MLP to Temporal LSTM
* Bullet Points:
  * Limitation of MLP: Evaluated only 1 static frame; blind to rate of turn, climb, or acceleration.
  * LSTM Advantage: Reads 10 consecutive seconds of flight data (10 frames x 14 features).
  * Training: 25 epochs on 15,739 sequences.
  * Test Accuracy: 98.30% (+31.5% improvement over baseline).
  * Test Precision: 98.44% | Test Recall: 99.02% (catches 2,841 out of 2,869 conflicts).
* Speaker Note:
  "Aircraft movement is dynamic, not a still photo. We upgraded from an MLP to an LSTM network that reads 10 seconds of trajectory history. This improved our conflict recall to 99.02% on unseen flights."

### Slide 4: Airspace Partitioning with Uber H3 Grid
* Slide Title: Spatial Indexing via Uber H3 Hexagons
* Bullet Points:
  * Avoids slow O(N^2) pairwise comparisons.
  * Airspace partitioned into Resolution 6 hexagons (approx 7.4 km diameter, 36.4 sq km area).
  * 1-ring neighborhood search covers approx 18 km, safely enclosing the FAA 5.0 nm (9.26 km) conflict boundary.
  * Distant aircraft pairs are pruned immediately, reducing compute overhead by 60% to 90%.
* Visual to Show: h3_grid_visualization.png
* Speaker Note:
  "We divide the airspace into Resolution 6 hexagons. Each hexagon is 7.4 km wide. By checking only flights within the same or neighboring hexagons, we safely enforce the 5-mile safety boundary while discarding distant flights."

### Slide 5: Uncertainty Modeling and Monte Carlo Simulation
* Slide Title: Sensor Noise Modeling and Monte Carlo Simulation
* Bullet Points:
  * Aviation Sensor Noise (DO-260B): +/- 100 m horizontal GPS drift, +/- 100 ft altitude error, +/- 5 km/h velocity noise.
  * Monte Carlo: Runs 100 to 500 stochastic simulations per pair with randomized noise.
  * Why it matters: A rigid CPA check only gives True or False. Monte Carlo reveals true statistical probability.
  * Example: Two aircraft at 950 ft vertical separation yield a 63% risk probability under sensor noise.
* Speaker Note:
  "Real ADS-B broadcasts suffer from GPS jitter and altimeter drift. Monte Carlo runs hundreds of randomized simulations per pair, turning a rigid yes/no answer into an actionable risk probability."

### Slide 6: Multi-Method Fusion and Dashboard
* Slide Title: Decision Fusion and Air Traffic Control Dashboard
* Bullet Points:
  * Fused Decision Formula: Final Risk = (0.4 * LSTM) + (0.6 * Monte Carlo).
  * Tiers: High Risk (>= 70%), Medium Risk (30% to 69%), Low Risk (< 30%).
  * Real-time Interactive Web Dashboard: Built with Flask, featuring live scenario selection and risk cards.
  * Visualizations: Trajectory conflict bubble plots and separation distance charts.
* Visual to Show: pipeline_conflict_result.png or Dashboard screenshot
* Speaker Note:
  "Our final decision engine combines the LSTM sequence model with Monte Carlo simulation. Air traffic controllers can monitor flights in real time through our interactive web dashboard."

---

## 7. Runnable Commands Summary

1. Run the Complete 5-Stage Demo:
   python demo_all.py

2. Inspect H3 Hexagonal Grid on Flight 78:
   python h3_spatial.py

3. Run Conflict Probability Self-Test:
   python conflict_probability.py

4. Run Full Pipeline on File 78:
   python run_full_pipeline.py --file test/78.txt

5. Launch the Web Dashboard:
   python dashboard.py
   (Then open http://127.0.0.1:5000 in your web browser)
