

from src.agents.reranker_agent import rerank_by_gemini

query = "How can nonthermal plasma, cold plasma or non-equilibrium plasma be applied for hair loss and hair-dye and hair removal"
doc = "A cold plasma helmet application device for delivery of cold plasma benefits to the head of a patient. An appropriate gas is introduced into a helmet receptacle within a containment dome of the helmet. The gas is energized by one or more dielectric barrier devices that receive energy from a pulsed source. The dielectric barrier devices can be configured to match the treatment area. Such a device and method can be used to treat large surface areas treatment sites associated with the head, head trauma, brain cancer, the control of brain swelling with closed head injury or infection, as well as treating male pattern baldness.Original"

print(rerank_by_gemini(query, doc))