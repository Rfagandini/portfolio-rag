"""
100 test questions for evaluating the RAG pipeline.
Each question has an expected answer (ground truth) for comparison.
Questions are grouped by document + include cross-document and follow-up questions.
"""

test_questions = [
    # === ALEXNET PAPER (1-25) ===

    # Basic facts
    {"id": 1, "input": "Who are the authors of the AlexNet paper?", "expected": "Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton"},
    {"id": 2, "input": "What university were the AlexNet authors from?", "expected": "University of Toronto"},
    {"id": 3, "input": "How many images were in the ImageNet LSVRC-2010 dataset?", "expected": "1.2 million"},
    {"id": 4, "input": "How many classes did AlexNet classify images into?", "expected": "1000 classes"},
    {"id": 5, "input": "What were AlexNet's top-1 and top-5 error rates on ILSVRC-2010?", "expected": "37.5% top-1 and 17.0% top-5"},
    {"id": 6, "input": "How many parameters does the AlexNet neural network have?", "expected": "60 million parameters"},
    {"id": 7, "input": "How many neurons does AlexNet have?", "expected": "650,000 neurons"},
    {"id": 8, "input": "How many convolutional layers does AlexNet have?", "expected": "Five convolutional layers"},
    {"id": 9, "input": "How many fully-connected layers does AlexNet have?", "expected": "Three fully-connected layers"},
    {"id": 10, "input": "What activation function did AlexNet use instead of tanh?", "expected": "ReLU (Rectified Linear Units)"},

    # Architecture details
    {"id": 11, "input": "How much faster did ReLU networks train compared to tanh on CIFAR-10?", "expected": "Six times faster"},
    {"id": 12, "input": "How many GPUs was AlexNet trained on?", "expected": "Two GPUs (NVIDIA GTX 580 3GB)"},
    {"id": 13, "input": "How long did it take to train AlexNet?", "expected": "Five to six days"},
    {"id": 14, "input": "How many cycles through the training set was AlexNet trained for?", "expected": "Roughly 90 cycles"},
    {"id": 15, "input": "What technique did AlexNet use to reduce overfitting with image augmentation?", "expected": "PCA-based color augmentation on RGB pixel values"},

    # Follow-up questions (testing conversational memory)
    {"id": 16, "input": "What was the previous state-of-the-art top-1 error before AlexNet?", "expected": "47.1% (sparse coding approach)"},
    {"id": 17, "input": "And the top-5?", "expected": "28.2%"},
    {"id": 18, "input": "What size were the kernels in AlexNet's second convolutional layer?", "expected": "5x5x48"},
    {"id": 19, "input": "How many kernels did that layer have?", "expected": "256 kernels"},

    # Deeper understanding
    {"id": 20, "input": "What is local response normalization in AlexNet?", "expected": "A normalization scheme that normalizes neuron activity using adjacent kernel maps at the same spatial position"},
    {"id": 21, "input": "Why did AlexNet need GPUs for training?", "expected": "CNNs were prohibitively expensive to apply to high-resolution images, and GPUs with optimized 2D convolution made training large CNNs feasible"},
    {"id": 22, "input": "What competition did AlexNet participate in?", "expected": "ILSVRC-2010 and ILSVRC-2012 (ImageNet Large Scale Visual Recognition Challenge)"},
    {"id": 23, "input": "What is the input dimensionality of AlexNet?", "expected": "150,528-dimensional"},
    {"id": 24, "input": "What dropout rate was used in AlexNet?", "expected": "The paper uses dropout in the fully-connected layers to reduce overfitting"},
    {"id": 25, "input": "What was the learning rate schedule used in AlexNet training?", "expected": "The learning rate was reduced three times prior to termination"},

    # === ATTENTION IS ALL YOU NEED (26-55) ===

    # Basic facts
    {"id": 26, "input": "What is the name of the model introduced in 'Attention Is All You Need'?", "expected": "The Transformer"},
    {"id": 27, "input": "Who are the authors of the Attention Is All You Need paper?", "expected": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin"},
    {"id": 28, "input": "What organization were most of the Transformer authors from?", "expected": "Google Brain and Google Research"},
    {"id": 29, "input": "What type of architecture did the Transformer replace?", "expected": "Recurrent neural networks (RNNs), specifically LSTM and gated recurrent networks"},
    {"id": 30, "input": "How many layers does the Transformer encoder have?", "expected": "N = 6 identical layers"},

    # Attention mechanism
    {"id": 31, "input": "What are the three components of the attention function?", "expected": "Queries, keys, and values"},
    {"id": 32, "input": "What type of attention does the Transformer use?", "expected": "Scaled Dot-Product Attention"},
    {"id": 33, "input": "Why is the dot product scaled in the attention mechanism?", "expected": "Divided by square root of dk to prevent dot products from growing too large"},
    {"id": 34, "input": "How many attention heads does the Transformer use?", "expected": "h = 8 parallel attention heads"},
    {"id": 35, "input": "What is multi-head attention?", "expected": "Multiple attention layers running in parallel, allowing the model to attend to information from different representation subspaces at different positions"},

    # Architecture
    {"id": 36, "input": "What are the two sub-layers in each Transformer encoder layer?", "expected": "Multi-head self-attention mechanism and a position-wise fully connected feed-forward network"},
    {"id": 37, "input": "What connection type is used around each sub-layer?", "expected": "Residual connections"},
    {"id": 38, "input": "Why does the Transformer need positional encoding?", "expected": "Because the model contains no recurrence and no convolution, so it needs positional encoding to use information about the order of the sequence"},
    {"id": 39, "input": "What is the complexity per layer of self-attention?", "expected": "O(n^2 * d) where n is sequence length and d is representation dimension"},
    {"id": 40, "input": "What is the complexity per layer of recurrent layers?", "expected": "O(n * d^2)"},

    # Results
    {"id": 41, "input": "What BLEU score did the Transformer achieve on English-to-German translation?", "expected": "The base model achieved 25.8 BLEU, and they outperformed previous models"},
    {"id": 42, "input": "What was the dimension of the Transformer base model?", "expected": "d_model = 512"},
    {"id": 43, "input": "What is the feed-forward dimension in the base Transformer?", "expected": "d_ff = 2048"},
    {"id": 44, "input": "What dropout rate was used in the Transformer?", "expected": "P_drop = 0.1"},
    {"id": 45, "input": "How many parameters does the base Transformer model have?", "expected": "65 million parameters"},

    # Follow-ups
    {"id": 46, "input": "What is the maximum path length for self-attention?", "expected": "O(1)"},
    {"id": 47, "input": "And for recurrent layers?", "expected": "O(n)"},
    {"id": 48, "input": "What benchmark was used for English-to-French translation?", "expected": "newstest2014"},
    {"id": 49, "input": "What task besides translation was the Transformer tested on?", "expected": "English constituency parsing"},
    {"id": 50, "input": "How did it perform on that task?", "expected": "Achieved 91.3 F1 on WSJ Section 23 with 4 layers"},

    # Conceptual
    {"id": 51, "input": "What advantage does self-attention have over recurrence for long sequences?", "expected": "Constant maximum path length O(1) vs O(n), meaning any two positions can be connected directly regardless of distance"},
    {"id": 52, "input": "What is restricted self-attention?", "expected": "Self-attention limited to considering only a neighborhood of size r in the input sequence, which increases path length to O(n/r)"},
    {"id": 53, "input": "How many training steps were used for the base Transformer?", "expected": "100K training steps"},
    {"id": 54, "input": "What dimensions are dk and dv in the base model?", "expected": "dk = 64 and dv = 64"},
    {"id": 55, "input": "What label smoothing value was used?", "expected": "epsilon_ls = 0.1"},

    # === IPCC CLIMATE REPORT (56-80) ===

    {"id": 56, "input": "What is the full title of the IPCC 2023 report?", "expected": "Climate Change 2023 Synthesis Report, Summary for Policymakers"},
    {"id": 57, "input": "Who is the chair of the IPCC report?", "expected": "Hoesung Lee"},
    {"id": 58, "input": "What percentage of global emissions came from energy, industry, transport, and buildings combined?", "expected": "The text states these sectors together accounted for emissions, with 22% from AFOLU (agriculture, forestry and other land use)"},
    {"id": 59, "input": "What does AFOLU stand for?", "expected": "Agriculture, Forestry and Other Land Use"},
    {"id": 60, "input": "What percentage of emissions came from AFOLU?", "expected": "22%"},

    {"id": 61, "input": "What is the remaining carbon budget for limiting warming to 1.5 degrees?", "expected": "If annual CO2 emissions between 2020-2030 stay at 2019 levels, cumulative emissions would almost exhaust the remaining carbon budget for 1.5C (50%)"},
    {"id": 62, "input": "What warming scenario does category C1 describe?", "expected": "Limit warming to 1.5C (>50%) with no or limited overshoot"},
    {"id": 63, "input": "What SSP scenario corresponds to category C1?", "expected": "Very low (SSP1-1.9)"},
    {"id": 64, "input": "What does SSP1-2.6 correspond to?", "expected": "Low emissions scenario"},
    {"id": 65, "input": "What warming does category C3 limit to?", "expected": "Limit warming to 2C (>67%)"},

    {"id": 66, "input": "What risks are projected to increase in the near term for all regions?", "expected": "Heat-related mortality and morbidity, food-borne, water-borne, and vector-borne diseases"},
    {"id": 67, "input": "What does climate resilient development mean according to the IPCC?", "expected": "The process of implementing greenhouse gas mitigation and adaptation measures to support sustainable development"},
    {"id": 68, "input": "What does the IPCC say about the window of opportunity for climate action?", "expected": "There is a rapidly narrowing window of opportunity to secure a liveable and sustainable future for all"},
    {"id": 69, "input": "What is needed to achieve climate goals regarding finance?", "expected": "Both adaptation and mitigation financing would need to increase many-fold; there is sufficient global capital but barriers to redirect it to climate action"},
    {"id": 70, "input": "What role does technology play according to the IPCC?", "expected": "Enhancing technology innovation systems is key to accelerate the widespread adoption of technologies and practices"},

    # Follow-ups
    {"id": 71, "input": "What about international cooperation?", "expected": "Finance, technology and international cooperation are critical enablers for accelerated climate action"},
    {"id": 72, "input": "How can ocean ecosystems help with climate change?", "expected": "Conservation and restoration of ocean ecosystems reduces vulnerability of biodiversity, reduces coastal erosion and flooding, and could increase carbon uptake if warming is limited"},
    {"id": 73, "input": "What does rebuilding fisheries do for climate adaptation?", "expected": "Reduces negative climate change impacts on fisheries and supports food security, biodiversity, human health and well-being"},
    {"id": 74, "input": "What has driven increases in emissions despite efficiency improvements?", "expected": "Rising global activity levels in industry, energy supply, transport, agriculture and buildings"},
    {"id": 75, "input": "What improvements reduced CO2 emissions from fossil fuels?", "expected": "Improvements in energy intensity of GDP and carbon intensity of energy"},

    {"id": 76, "input": "What confidence level does the IPCC assign to near-term hazard increases?", "expected": "Medium to high confidence depending on region and hazard"},
    {"id": 77, "input": "What does the IPCC say about existing fossil fuel infrastructure?", "expected": "Projected CO2 emissions from existing fossil fuel infrastructure without additional abatement already exceed the remaining carbon budget for 1.5C (50%)"},
    {"id": 78, "input": "What is RCP 4.5 associated with?", "expected": "Intermediate emissions scenario (SSP2-4.5)"},
    {"id": 79, "input": "What warming category is associated with RCP 8.5?", "expected": "Higher warming categories (C6-C7 range)"},
    {"id": 80, "input": "What year range does the carbon budget analysis focus on?", "expected": "2020-2030"},

    # === NASA ARTEMIS PLAN (81-100) ===

    {"id": 81, "input": "What is NASA's Artemis program?", "expected": "NASA's lunar exploration program to return humans to the Moon and prepare for Mars exploration"},
    {"id": 82, "input": "When was the Artemis plan published?", "expected": "September 2020"},
    {"id": 83, "input": "Who wrote the foreword of the Artemis plan?", "expected": "Jim Bridenstine (NASA Administrator)"},
    {"id": 84, "input": "What is Artemis named after?", "expected": "The twin sister of Apollo, Goddess of the Moon"},
    {"id": 85, "input": "What was the deadline set for landing astronauts on the Moon?", "expected": "2024, four years earlier than originally planned"},

    {"id": 86, "input": "Who challenged NASA to accelerate the Moon landing timeline?", "expected": "Vice President Pence"},
    {"id": 87, "input": "What three companies were selected for the Human Landing System?", "expected": "Blue Origin, Dynetics (a Leidos company), and SpaceX"},
    {"id": 88, "input": "Where is Blue Origin based?", "expected": "Kent, Washington"},
    {"id": 89, "input": "Where is SpaceX based?", "expected": "Hawthorne, California"},
    {"id": 90, "input": "Where is Dynetics based?", "expected": "Huntsville, Alabama"},

    {"id": 91, "input": "What is the Gateway in the Artemis program?", "expected": "A space station in lunar orbit that will establish U.S. presence in the region between Earth and the Moon"},
    {"id": 92, "input": "What is VIPER?", "expected": "Volatiles Investigating Polar Exploration Rover, managed by Ames Research Center"},
    {"id": 93, "input": "What does SLS stand for in the Artemis program?", "expected": "Space Launch System"},
    {"id": 94, "input": "What spacecraft carries astronauts in the Artemis program?", "expected": "Orion"},
    {"id": 95, "input": "How many RS-25 engines does the SLS have?", "expected": "Four RS-25 liquid rocket engines"},

    {"id": 96, "input": "What are the desired traits for lunar landing sites?", "expected": "Access to significant sunlight, continuous line-of-sight to Earth, mild grading and surface debris for safe landing, and proximity to permanently shadowed regions"},
    {"id": 97, "input": "What is NASA's long-term goal beyond the Moon?", "expected": "Human exploration of Mars"},
    {"id": 98, "input": "What does NASA plan regarding lunar resources?", "expected": "Establish that lunar resources can be extracted and purchased from the private sector in compliance with the Outer Space Treaty"},
    {"id": 99, "input": "What is the role of international partnerships in Artemis?", "expected": "NASA is building a coalition of partnerships with industry, nations and academia to get to the Moon quickly and sustainably"},
    {"id": 100, "input": "What mission comes after Artemis III?", "expected": "Missions extending lunar presence and preparing for Mars, establishing infrastructure and sustained surface presence"},
]


# Pairs where q[i+1] is a follow-up to q[i] (testing conversational memory)
follow_up_pairs = [
    (16, 17),   # AlexNet previous SOTA top-1 -> and top-5?
    (18, 19),   # AlexNet kernel size -> how many kernels?
    (46, 47),   # Transformer max path self-attention -> and recurrent?
    (49, 50),   # Transformer other task -> how did it perform?
    (70, 71),   # IPCC technology role -> what about international cooperation?
]
