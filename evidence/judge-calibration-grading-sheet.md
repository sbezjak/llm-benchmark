# Judge calibration - grade these yourself, blind

You are the ground truth. Grade each answer **without** knowing what the model judge (gpt-5.6-luna) gave it - that is the whole point. Afterwards we compare your grades to the judge's and report how often they agree.

**How to grade** - for each answer, mark one:

- **Good** - correct and answers the question well
- **Weak** - partly right, missing something, or flawed but not wrong
- **Wrong** - incorrect, off-topic, or unusable

Put an `x` in one box. Add a short note if you want (optional). Do **not** open `evidence/judge-calibration-key.json` until you are done.

Sample: **24 answers** (14 the judge scored below perfect, the rest it scored perfect - you don't get told which, or in what order).

> These are the 2026-08-17 re-roll answers. Regenerate this sheet with `python -m llm_benchmark.calibration` (pure, $0).

---

### G01 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
Let's say the ball costs x dollars.

Since the bat costs $1.00 more than the ball, the bat costs x + $1.00.

The total cost of the bat and the ball is $1.10, so we can set up an equation:

x + (x + $1.00) = $1.10

Combine like terms:

2x + $1.00 = $1.10

Subtract $1.00 from both sides:

2x = $0.10

Divide by 2:

x = $0.05

So, the ball costs $0.05.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

Could be simpler

### G02 · item `reasoning_003`

**Question:** How does photosynthesis work in plants?

**Expected:** Photosynthesis in plants converts light energy into chemical energy, using sunlight to transform carbon dioxide and water into glucose and oxygen.

**Answer:**

~~~~
Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy in the form of organic compounds, such as glucose. This process is essential for life on Earth, as it provides energy and organic compounds for food chains.

Here's a step-by-step explanation of how photosynthesis works:

1. **Light absorption**: Light-dependent reactions occur when light is absorbed by pigments such as chlorophyll, which is present in the thylakoid membranes of chloroplasts (organelles found in plant cells).
2. **Excitation of electrons**: When light energy is absorbed, it excites electrons in the pigment molecules, which are then transferred to a special molecule called an electron acceptor.
3. **Transfer of electrons**: The electrons are passed along a series of electron carriers in the thylakoid membrane, ultimately resulting in the formation of a high-energy molecule called ATP (adenosine triphosphate).
4. **Water splitting**: Water molecules (H2O) are split into oxygen (O2), protons (H+), and electrons using energy from light.
5. **Calvin cycle**: The ATP and NADPH produced in the light-dependent reactions are used to power the Calvin cycle, also known as the light-independent reaction or carbon fixation.
6. **Carbon fixation**: CO2 is fixed into a three-carbon molecule called 3-phosphoglycerate (3-PGA) using energy from ATP and NADPH.
7. **Reduction of 3-PGA**: The 3-PGA molecules are reduced to form glyceraldehyde-3-phosphate (G3P), which is then used to synthesize glucose.

The overall equation for photosynthesis is:

6 CO2 + 6 H2O + light energy → C6H12O6 (glucose) + 6 O2

In summary, photosynthesis involves the conversion of light energy into chemical energy through a series of light-dependent and light-independent reactions. This process is essential for plant growth and development, as well as supporting life on Earth.

**Key players in photosynthesis:**

* Chlorophyll (pigment)
* Thylakoid membranes
* Chloroplasts (organelles)
* ATP (energy molecule)
* NADPH (reduction agent)

**Interesting facts about photosynthesis:**

* Photosynthesis occurs in specialized organelles called chloroplasts, which contain pigments such as chlorophyll.
* The process of photosynthesis is responsible for producing approximately 70% of the oxygen we breathe.
* Plants can undergo a process called crassulacean acid metabolism (CAM) to conserve water and optimize photosynthesis in arid environments.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G03 · item `adv_logic_001`

**Question:** Mango is heavier than Kiwi but lighter than Pomelo. Pomelo is lighter than Durian. Of the four, which is the heaviest?

**Expected:** Durian

**Answer:**

~~~~
To determine which fruit is the heaviest, let's analyze the given information:

1. Mango is heavier than Kiwi.
2. Mango is lighter than Pomelo.
3. Pomelo is lighter than Durian.

From statement 1 and 2, we can conclude that Durian is heavier than Mango (since Mango is lighter than Pomelo).

Now, from statements 1-3, we have:

Durian > Mango
Mango > Kiwi

So, Durian is the heaviest fruit among the four.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G04 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
The ball costs **$0.05** (5 cents).

Let the ball cost \(x\). Then the bat costs \(x + \$1.00\):

\[
x + (x + 1.00) = 1.10
\]

\[
2x = 0.10 \quad\Rightarrow\quad x = 0.05
\]
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

Could be simpler

### G05 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
The ball costs **5 cents**.

Explanation: If the ball costs $0.05, the bat costs $0.05 + $1.00 = $1.05. Together, $1.05 + $0.05 = $1.10.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G06 · item `adv_riddle_001`

**Question:** A red house is made of red bricks and a blue house is made of blue bricks. What is a greenhouse made of?

**Expected:** Glass

**Answer:**

~~~~
That's a clever trick question!

The answer isn't necessarily related to the color of the houses, but rather to the fact that greenhouses are typically made of glass or plastic materials, not bricks.

So, while the question tries to distract us with the colors of the houses, the correct answer is simply "glass" or "plastic", depending on the type of greenhouse.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G07 · item `procedural_001`

**Question:** Which pytest command runs tests marked either 'slow' or 'network'?

**Expected:** pytest -m 'slow or network'

**Answer:**

~~~~
You can use the `-k` option with pytest to run tests that match a specific marker. In this case, you would use:

```bash
pytest -k "slow" --network
```

However, since `--network` is not a valid option for pytest, it's likely that you meant to use the `-m` option instead.

If you want to run tests marked with either 'slow' or 'network', you can use:

```bash
pytest -m "slow" -m "network"
```

This will run all tests that are marked with either 'slow' or 'network'.
~~~~

- [ ] Good   - [ ] Weak   - [x] Wrong

Note: 

---

### G08 · item `reasoning_003`

**Question:** How does photosynthesis work in plants?

**Expected:** Photosynthesis in plants converts light energy into chemical energy, using sunlight to transform carbon dioxide and water into glucose and oxygen.

**Answer:**

~~~~
Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy in the form of organic compounds, such as glucose. This process is essential for life on Earth, as it provides energy and organic compounds for plants to grow and thrive.

Here's a step-by-step explanation of how photosynthesis works:

**The Light-Dependent Reactions**

1. **Light absorption**: Light is absorbed by pigments such as chlorophyll and other accessory pigments in the thylakoid membranes of chloroplasts.
2. **Excitation of electrons**: The absorbed light energy excites electrons, which are then transferred to a special molecule called an electron acceptor.
3. **Transfer of electrons**: The electrons are passed along a series of electron carriers in the thylakoid membrane, ultimately resulting in the formation of a high-energy molecule called ATP (adenosine triphosphate).
4. **Production of NADPH**: Another energy-rich molecule called NADPH is also produced during this stage.

**The Light-Independent Reactions**

1. **Carbon fixation**: Carbon dioxide from the atmosphere is fixed into a three-carbon molecule called 3-phosphoglycerate (3-PGA) via the enzyme RuBisCO.
2. **Reduction of CO2**: The 3-PGA molecules are reduced to form glyceraldehyde-3-phosphate (G3P) using the energy from ATP and NADPH produced in the light-dependent reactions.
3. **Regeneration of RuBP**: The enzyme RuBP is regenerated, allowing the cycle to continue.

**Overall Equation**

6 CO2 + 6 H2O + light energy → C6H12O6 (glucose) + 6 O2

In summary, photosynthesis involves the conversion of light energy into chemical energy through a series of light-dependent and light-independent reactions. The overall equation shows that six carbon dioxide molecules and six water molecules are converted into one glucose molecule and six oxygen molecules.

**Importance of Photosynthesis**

Photosynthesis is essential for life on Earth because it:

1. Provides energy and organic compounds for plants to grow and thrive.
2. Supports the food chain by producing the base of the food web (plants).
3. Produces oxygen, which is necessary for the survival of most living organisms.

In summary, photosynthesis is a complex process that involves the conversion of light energy into chemical energy, providing the energy and organic compounds needed for plant growth and supporting life on Earth.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G09 · item `reasoning_001`

**Question:** What is the difference between cold blooded and warm blooded animals?

**Expected:** Cold blooded animals rely on external sources for body heat, while warm blooded animals maintain a constant internal body temperature.

**Answer:**

~~~~
The terms "cold-blooded" and "warm-blooded" are outdated and can be misleading. The correct terminology is "ectothermic" (cold-blooded) and "endothermic" (warm-blooded).

**Ectothermic (Cold-Blooded) Animals:**

Ectothermic animals, also known as cold-blooded animals, rely on external sources of heat to regulate their body temperature. Their metabolic rate is influenced by the environment around them, and they often bask in sunlight or seek shade to maintain a stable body temperature.

Examples of ectothermic animals include:

* Reptiles (snakes, lizards, turtles)
* Amphibians (frogs, toads)
* Fish
* Insects (many species)

**Endothermic (Warm-Blooded) Animals:**

Endothermic animals, also known as warm-blooded animals, generate heat internally and maintain a relatively constant body temperature regardless of the environment. This is made possible by their ability to produce energy through cellular respiration.

Examples of endothermic animals include:

* Mammals (humans, dogs, cats)
* Birds
* Some species of fish (such as goldfish)

Key differences between ectothermic and endothermic animals:

1. **Body temperature regulation:** Ectothermic animals rely on external sources of heat, while endothermic animals generate heat internally.
2. **Metabolic rate:** Endothermic animals have a higher metabolic rate than ectothermic animals, which allows them to maintain their body temperature.
3. **Energy production:** Endothermic animals produce energy through cellular respiration, while ectothermic animals rely on external sources of energy (e.g., food).
4. **Activity patterns:** Ectothermic animals often have more restricted activity patterns, as they need to regulate their body temperature. Endothermic animals can be active at a wider range of temperatures.

It's worth noting that the terms "cold-blooded" and "warm-blooded" were historically used to describe these two groups, but they are now considered outdated and can be misleading. The correct terminology is ectothermic and endothermic.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G10 · item `adv_count_001`

**Question:** How many times does the letter 'r' appear in the word 'strawberry'?

**Expected:** 3

**Answer:**

~~~~
The letter "r" appears twice in the word "strawberry".
~~~~

- [ ] Good   - [ ] Weak   - [x] Wrong

Note: 

---

### G11 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
Let's call the cost of the ball "x".

Since the bat costs $1.00 more than the ball, the cost of the bat is x + $1.00.

The total cost of both the bat and the ball together is $1.10. We can set up an equation to represent this:

x (ball) + (x + $1.00) (bat) = $1.10

Combine like terms:

2x + $1.00 = $1.10

Subtract $1.00 from both sides:

2x = $0.10

Divide by 2:

x = $0.05

So, the ball costs $0.05 (or 5 cents).
~~~~

- [x] Good   - [] Weak   - [ ] Wrong

Note: 

Could be simpler

### G12 · item `adv_logic_001`

**Question:** Mango is heavier than Kiwi but lighter than Pomelo. Pomelo is lighter than Durian. Of the four, which is the heaviest?

**Expected:** Durian

**Answer:**

~~~~
To determine the heaviest fruit among the four, let's analyze the given information:

1. Mango is heavier than Kiwi.
2. Mango is lighter than Pomelo.
3. Pomelo is lighter than Durian.

From statement 1 and 2, we can conclude that Durian is heavier than Mango (since Mango is both heavier than Kiwi and lighter than Pomelo).

Now, let's look at the relationships between Pomelo and Durian:

- Pomelo is lighter than Durian.
- We already know that Durian is heavier than Mango.

Since Durian is heavier than Mango, and Pomelo is lighter than Durian, it means that Durian must be heavier than Pomelo as well.

So, based on the given information, we can conclude that:

Durian > Mango > Pomelo > Kiwi

Therefore, Durian is the heaviest fruit among the four.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G13 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
# Solution

Let me set up equations where:
- b = cost of the ball
- B = cost of the bat

**Given information:**
- b + B = $1.10
- B = b + $1.00

**Solving:**

Substitute the second equation into the first:

b + (b + $1.00) = $1.10

2b + $1.00 = $1.10

2b = $0.10

**b = $0.05**

The ball costs **$0.05** (5 cents).

(The bat costs $1.05)
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

Could be simpler

### G14 · item `procedural_001`

**Question:** Which pytest command runs tests marked either 'slow' or 'network'?

**Expected:** pytest -m 'slow or network'

**Answer:**

~~~~
You can use the following pytest command to run tests marked with either 'slow' or 'network':

```bash
pytest --durations=1s --filter='slow|network'
```

This will run all tests that are marked with 'slow' or 'network', and it will only consider tests that take less than 1 second.
~~~~

- [ ] Good   - [ ] Weak   - [x] Wrong

Note: 

---

### G15 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
# Bat and Ball Problem

**Answer: $0.05 (5 cents)**

## The Math

Let the ball cost = *x*

Then the bat costs = *x* + $1.00

Together: 
$$x + (x + 1.00) = 1.10$$
$$2x + 1.00 = 1.10$$
$$2x = 0.10$$
$$x = 0.05$$

## Verification
- Ball: $0.05
- Bat: $0.05 + $1.00 = $1.05
- Total: $0.05 + $1.05 = **$1.10** ✓
- Bat costs $1.00 more than ball: $1.05 − $0.05 = **$1.00** ✓

## Why This Trips People Up

Most people's intuition jumps to **$0.10** because $1.00 + $0.10 = $1.10 total. But that's incorrect—if the ball were $0.10, the bat would need to be $1.00 *more* than that, making it $1.10, for a total of $1.20 (not $1.10). This is a classic example from behavioral economics (popularized by Daniel Kahneman) showing how our fast, intuitive thinking can lead us astray on problems that seem simple at first glance.
~~~~

- [x] Good   - [] Weak   - [ ] Wrong

Note: 

Could be simpler

### G16 · item `adv_factual_002`

**Question:** Which is the only sea on Earth with no land coastline, bounded instead by ocean currents?

**Expected:** The Sargasso Sea

**Answer:**

~~~~
The answer is the Mediterranean Sea. It is a semi-enclosed sea that is connected to the Atlantic Ocean and the Black Sea via various straits. The Mediterranean Sea does not have any direct land coastline, but it is surrounded by coastlines of countries such as Greece, Turkey, Italy, Spain, France, and others.

However, I should note that some sources may argue that the Caspian Sea could also be considered a sea with no land coastline, as it is technically a lake (the largest inland body of water) rather than an ocean. But among the world's oceans, the Mediterranean Sea is indeed the one with no direct land coastline.

Please let me know if you have any further questions!
~~~~

- [ ] Good   - [ ] Weak   - [x] Wrong

Note: 

---

### G17 · item `adv_riddle_001`

**Question:** A red house is made of red bricks and a blue house is made of blue bricks. What is a greenhouse made of?

**Expected:** Glass

**Answer:**

~~~~
A greenhouse is typically made mostly of **glass (or clear plastic/polycarbonate) panels supported by a frame** of wood, aluminum, or steel — not green bricks!

The name is a bit of a linguistic trick, similar to your examples. Unlike a "red house" or "blue house," where the color describes the building material, a "greenhouse" gets its name from its *purpose*: it's a structure designed to grow green plants by trapping sunlight and heat inside, creating a warm, controlled environment.

So the "green" in greenhouse refers to the plants grown inside, not the color of the structure itself. It's a classic example of how English compound words don't always follow logical patterns — the same trick often used in riddles like "Why do they call it a hamburger when it's made of beef?"
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G18 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
# Finding the Ball's Cost

Let me set up equations where:
- b = cost of the ball
- B = cost of the bat

**From the problem:**
1) b + B = 1.10
2) B = b + 1.00

**Substituting equation 2 into equation 1:**
- b + (b + 1.00) = 1.10
- 2b + 1.00 = 1.10
- 2b = 0.10
- b = 0.05

**The ball costs $0.05** (5 cents)

To verify: The bat costs $1.05, and together $1.05 + $0.05 = $1.10 ✓
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G19 · item `adv_factual_002`

**Question:** Which is the only sea on Earth with no land coastline, bounded instead by ocean currents?

**Expected:** The Sargasso Sea

**Answer:**

~~~~
The answer is the Mediterranean Sea. It is a semi-enclosed sea that is connected to the Atlantic Ocean and the Black Sea via various straits. The Mediterranean Sea does not have any direct land coastline; instead, it is surrounded by the coasts of Europe, Africa, and Asia, which are bounded by ocean currents.
~~~~

- [ ] Good   - [ ] Weak   - [x] Wrong

Note: 

---

### G20 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
# Bat and Ball Problem

**Answer: $0.05 (5 cents)**

**Why it's not $0.10:**
The intuitive (but wrong) answer is that the ball costs $0.10, but that would make the bat $1.10, for a total of $1.20 — too much.

**The correct math:**

Let the ball = *x*
Then the bat = *x* + $1.00

Together: 
$$x + (x + 1.00) = 1.10$$
$$2x + 1.00 = 1.10$$
$$2x = 0.10$$
$$x = 0.05$$

**Check:**
- Ball: $0.05
- Bat: $0.05 + $1.00 = $1.05
- Total: $0.05 + $1.05 = **$1.10** ✓

This is a famous example from psychologist Daniel Kahneman's research, used to illustrate how our brains often jump to a fast, intuitive answer rather than pausing to reason through the problem carefully.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

Could be simpler

### G21 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
The ball costs **$0.05** (5 cents).

The bat costs $1.05, totaling $1.10, and it is exactly $1.00 more than the ball.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G22 · item `reasoning_001`

**Question:** What is the difference between cold blooded and warm blooded animals?

**Expected:** Cold blooded animals rely on external sources for body heat, while warm blooded animals maintain a constant internal body temperature.

**Answer:**

~~~~
The terms "cold-blooded" and "warm-blooded" are outdated and somewhat misleading, as they don't accurately describe an animal's metabolic processes. Instead, the correct terminology is "ectothermic" (cold-blooded) and "endothermic" (warm-blooded).

Here's a brief explanation of each:

**Ectothermic (Cold-Blooded):**

* Animals that regulate their body temperature using external sources, such as the sun or a warm rock.
* Their metabolic rate is influenced by the environment, and they often bask in sunlight to increase their body temperature.
* Examples: reptiles (e.g., snakes, lizards), amphibians (e.g., frogs, toads), fish.

**Endothermic (Warm-Blooded):**

* Animals that generate heat internally through metabolic processes, such as muscle contractions and cellular respiration.
* They maintain a relatively constant body temperature, regardless of the environment.
* Examples: mammals (e.g., humans, dogs, cats) and birds.

The key differences between ectothermic and endothermic animals are:

1. **Temperature regulation**: Ectotherms rely on external sources to regulate their body temperature, while endotherms generate heat internally.
2. **Metabolic rate**: Ectotherms have a slower metabolic rate than endotherms, as they don't need to burn energy to maintain a constant body temperature.
3. **Activity patterns**: Ectotherms often have more restricted activity patterns, as they may be active only when the environment is warm enough for them to regulate their body temperature.

It's worth noting that some animals can exhibit both ectothermic and endothermic characteristics, depending on the situation. For example, some reptiles can bask in sunlight to increase their body temperature, while others may seek shade to cool down.

In summary, the terms "cold-blooded" and "warm-blooded" are outdated and don't accurately describe an animal's metabolic processes. Instead, we use the more accurate terms "ectothermic" (cold-blooded) and "endothermic" (warm-blooded) to describe these differences.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G23 · item `procedural_001`

**Question:** Which pytest command runs tests marked either 'slow' or 'network'?

**Expected:** pytest -m 'slow or network'

**Answer:**

~~~~
# Running Tests Marked 'slow' or 'network'

To run pytest tests marked with either `slow` or `network` markers, use the `-m` flag with an OR expression:

```bash
pytest -m "slow or network"
```

## Additional Useful Variations

**Run tests with verbose output:**
```bash
pytest -m "slow or network" -v
```

**Run tests that are NOT slow or network** (opposite/exclude):
```bash
pytest -m "not slow and not network"
```

**Combine with other markers:**
```bash
pytest -m "(slow or network) and not skip_ci"
```

## Prerequisite: Register Custom Markers

If you haven't already, register these markers in your `pytest.ini`, `pyproject.toml`, or `setup.cfg` to avoid warnings:

**pytest.ini:**
```ini
[pytest]
markers =
    slow: marks tests as slow
    network: marks tests that require network access
```

**pyproject.toml:**
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "network: marks tests that require network access",
]
```

Then mark your tests like this:
```python
import pytest

@pytest.mark.slow
def test_something_slow():
    ...

@pytest.mark.network
def test_api_call():
    ...
```
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---

### G24 · item `adv_arith_001`

**Question:** A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

**Expected:** $0.05 (5 cents)

**Answer:**

~~~~
The ball costs **$0.05** (5 cents).

Explanation: If the ball costs \(x\), the bat costs \(x + \$1.00\). Together:

\[
x + (x + 1.00) = 1.10
\]

\[
2x = 0.10 \Rightarrow x = 0.05
\]

So the ball is **5 cents** and the bat is **$1.05**.
~~~~

- [x] Good   - [ ] Weak   - [ ] Wrong

Note: 

---
