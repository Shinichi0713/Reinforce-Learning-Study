
# What Is Physical AI?

Over the past two years, the concept of **Physical AI** has rapidly gained attention.
Although the term can be somewhat vague, this article summarizes what is currently understood about it.

---

## Overview

**Physical AI** refers to technologies in which **artificial intelligence possesses a physical body**—such as robots, drones, autonomous vehicles, or smart appliances—and operates autonomously in the real world.

Traditional AI (digital AI) primarily performs **information processing or content generation within digital environments**, such as text or image generation on screens.

In contrast, Physical AI performs a **continuous real-time cycle**:

1. Perceive the real-world environment
2. Make decisions
3. Physically act upon the environment

---

## Key Differences from Traditional AI

### Operational Environment

Traditional generative AI systems (such as ChatGPT) operate entirely within **digital space**.
Physical AI, on the other hand, operates within the **physical world**.

---

### Type of Data Processed

Traditional AI primarily handles **text and images**.

Physical AI processes **complex physical data obtained through sensors**, including:

* gravity
* friction
* distance
* obstacles
* temporal dynamics (time-series data)

---

### Autonomy of Behavior

Conventional robots typically perform **repetitive pre-programmed tasks**.

Physical AI systems instead **adapt their behavior to surrounding conditions**, autonomously determining optimal actions.

---

## Application Areas and Examples of Physical AI

### Manufacturing and Logistics

* General-purpose robots capable of grasping and transporting diverse products
* Automated transport systems inside factories (AGV / AMR)

---

### Hazardous Environments

Physical AI can operate in environments that are dangerous for humans:

* high-altitude operations
* fire sites
* hazardous material zones

---

### Humanoid Robots

Humanoid robots capable of natural conversation while performing household or light industrial tasks are under development by companies such as:

* Figure AI
* Tesla

---

## Why Physical AI Is Attracting Attention

The rapid growth of Physical AI is driven by several global trends:

* severe labor shortages
* continued digital transformation (DX)

Major technology companies such as:

* NVIDIA
* SoftBank Group

are investing massive amounts of capital.

Because of this rapid acceleration, **2026 is sometimes referred to as the “first year of Physical AI.”**

---

# Changes in Learning Paradigms

The learning targets of Physical AI differ dramatically from those of traditional digital AI.

Instead of learning from **screen-based information**, Physical AI learns from **physical interactions in the real world**.

---

## 1. From Digital Data to Multimodal Physical Data

Traditional AI models such as ChatGPT were trained primarily on **large amounts of internet text and images**.

Physical AI integrates multiple sources of sensory information obtained through a physical body.

Examples include:

### Vision and Spatial Perception

* cameras
* LiDAR

These provide **3D spatial understanding**, including distances and object placement.

---

### Force and Tactile Feedback

* pressure when grasping objects
* friction
* object weight

---

### Time-Series Motion Data

Physical actions involve time-dependent changes such as:

* motion timing
* acceleration
* velocity

---

## 2. From Static Knowledge to Dynamic Trial-and-Error

Traditional AI primarily performs **pattern inference from existing datasets**.

Physical AI instead emphasizes **reinforcement learning**, where the system repeatedly acts in the real world (or high-fidelity simulation environments) and learns through trial and error.

Example:

Learning how to **lift a cup without breaking it** requires discovering the appropriate amount of force through experience.

---

## Why Reinforcement Learning Is Necessary

Physical AI relies heavily on **reinforcement learning rather than supervised learning** because the real world rarely has a single correct labeled answer.

Physical interactions must be **experienced**.

---

### 1. Motion Cannot Be Fully Described with Labels

Traditional AI can learn from labeled images such as:

> "This is a cat."

However, if a robot must **pick up an egg without breaking it**, it is impossible to provide precise labeled data for:

* finger angles at millimeter precision
* exact gripping force
* slip conditions

Reinforcement learning instead provides a simple objective:

* reward if the egg is lifted without breaking

The AI then discovers the optimal behavior.

---

### 2. Adaptation to Unknown Environments

Real-world environments constantly change:

* floor friction
* wind strength
* object weight

If a robot only imitates pre-programmed movements, it fails when conditions change.

By learning **strategies** through reinforcement learning, the system can adapt to new environments and unexpected obstacles.

---

### 3. Continuous Decision-Making

Physical motion is a **continuous process**, where each decision affects the next state.

Example: Human walking.

After stepping forward with the right foot, the center of gravity shifts.

The next step depends on that shift.

Therefore, rather than evaluating each frame independently, reinforcement learning optimizes **the cumulative reward over an entire sequence of actions**.

---

## 3. From Language Rules to Understanding Physical Laws

Traditional digital AI learns:

* grammar
* logic

Physical AI must instead learn:

* gravity
* inertia
* material stiffness

This allows it to respond flexibly in unexpected situations such as:

* slippery floors
* sudden obstacles

---

## How Physical AI Learns Physical Laws

Two main approaches are currently used.

---

### 1. Training in Physics Simulators

Real-world trial-and-error can be dangerous and inefficient.

Instead, training occurs in simulated environments such as:

* NVIDIA Isaac
* digital twin systems

These environments simulate:

* gravity
* friction
* air resistance
* material hardness

AI systems can perform **millions of training iterations**, learning causal relationships such as:

* how objects bounce
* how gripping angles affect slipping

---

### 2. World Models

A major recent advancement is the development of **internal world models**.

A world model acts as an **internal simulator of reality**.

Example:

By observing large amounts of video where a ball follows a parabolic trajectory, the AI learns **physical intuition** without explicitly learning equations.

Before acting, the AI mentally simulates possible outcomes and selects the safest and most efficient action.

---

### 3. Integration with Large Language Models (LLMs)

Modern Physical AI systems integrate **large language models**.

Example instruction:

> "The sponge is soft, so hold it gently."

The AI interprets this conceptually:

* soft → easily deformed → reduce applied force

The system converts **linguistic knowledge into physical action**.

---

# Core Architecture of Physical AI

The architecture of Physical AI resembles a **multi-layer nervous system**.

LLMs function as part of the **brain**, but additional components act like:

* cerebellum
* spinal cord

The architecture typically contains three layers.

---

## 1. Vision-Language Models (VLM) — The “Cerebrum”

A **VLM** combines a large language model with visual perception.

Example base models:

* GPT-style LLMs with image input capability.

Role:

Understanding abstract instructions.

Example instruction:

> "Clean the messy room."

The system interprets camera input and produces a plan:

* trash → trash bin
* cup → table

---

## 2. World Models — Physical Intuition

LLMs alone cannot simulate physics.

A **world model** predicts physical consequences.

Example:

> "If I drop this cup, it will fall and break."

Technically, these models often adapt techniques from **video generation AI**, predicting future states of the environment.

---

## 3. Policy Networks — Motor Control

This is the lowest-level control layer responsible for moving motors.

Role:

Convert high-level instructions like

> "Pick up the cup"

into precise motor commands.

These commands control:

* force
* speed
* position

Reinforcement learning is heavily used in this layer.

---

# Major Players in Physical AI

## 1. Platform and Infrastructure Companies

These companies provide the **training environments and core AI models**.

### NVIDIA

Provides:

* Omniverse physics simulation platform
* humanoid robot foundation model **Project GR00T**

NVIDIA is currently the **central hub of the Physical AI ecosystem**.

---

### Amazon (AWS)

Provides cloud infrastructure for robotics development.

AWS also collaborates with NVIDIA to support Physical AI startups.

---

### SoftBank Group

SoftBank has invested heavily in **Skild AI**, which is developing a universal robot control AI called **Skild Brain**.

Their strategy is to integrate Physical AI with global communication networks.

---

## 2. Humanoid and General-Purpose Robot Developers

These companies develop both the **physical robot body** and the controlling AI.

### Tesla

Developing the humanoid robot **Optimus**.

Tesla uses driving data from millions of vehicles for AI training.

---

### Figure AI

Partnered with OpenAI.

Developing humanoid robots such as **Figure 03**, capable of conversation and physical tasks.

---

### Boston Dynamics

Now owned by Hyundai.

Their new electric **Atlas** robot is being introduced into industrial environments.

---

### 1X Technologies

A Norwegian robotics company backed by OpenAI.

Developing the home robot **NEO**.

---

## 3. Key Japanese Companies

Japan remains strong in **robot hardware and industrial robotics**.

Examples include:

* Telexistence
* FANUC
* Yaskawa Electric
* NEC
* Toyota

Toyota collaborates closely with NVIDIA on digital twin technology and autonomous driving.

---

## 4. Major Chinese Players

China may become one of the most significant sources of innovation.

Chinese companies include:

### Unitree Robotics

Produces affordable, high-performance humanoid and quadruped robots.

---

### Xiaomi

Developing humanoid robots such as **CyberOne**.

---

# Conclusion

This article summarized the current state of Physical AI.

---

## Architecture

Physical AI systems typically combine multiple AI architectures:

* VLM for perception and reasoning
* world models for physical prediction
* policy networks for motor control

---

## Learning Methods

Physical AI learns through:

* multimodal sensor data
* trial-and-error interaction with the environment

Unlike traditional AI, simply feeding information is not sufficient when interacting with the physical world.

---

## Industry Players

Major companies include:

* Amazon
* NVIDIA
* Tesla

Japan has companies such as:

* FANUC
* Yaskawa
* SoftBank

China also has strong players such as:

* Xiaomi
* Unitree Robotics

---

If Physical AI becomes fully realized, robots similar to **Doraemon** may one day move freely in the real world.

However, Japan may lag behind in innovation and adoption, despite being a country that urgently needs such technology due to its rapidly aging population.

This article aimed to organize and clarify the current state of Physical AI.

