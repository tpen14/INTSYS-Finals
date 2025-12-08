import logging
import base64
import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import aiohttp
import json
import re
from io import BytesIO
from PIL import Image
import ollama  # Uses the official Ollama Python client

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """
    Analyzes agricultural images to identify pests and diseases SPECIFIC TO THE PHILIPPINES.
    Uses vision APIs (Google Vision, Claude Vision, or Ollama multimodal).
    """

    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_vision_model = os.getenv("OLLAMA_VISION_MODEL", "llava")
        self.timeout = aiohttp.ClientTimeout(total=60)

        # Strictly Philippine Agricultural Pests
        self.pest_database = self._build_pest_database()
        self.disease_database = self._build_disease_database()

    def _build_pest_database(self) -> Dict:
        """Build Philippine agricultural pest database with CAR-specific pests (Verified List)"""
        return {
            "armyworm": {
                "scientific_name": "Spodoptera litura / frugiperda",
                "local_name": "Harabas / Uod / Army Worm",
                "crops_affected": ["rice", "corn", "onion", "cabbage", "lettuce"],
                "description": "Green to brown caterpillars with dark stripes. Feed on leaves leaving only veins. Common in CAR vegetable farms and rice terraces.",
                "visual_signs": ["Holes in leaves", "Leaf skeletonization", "Visible caterpillars", "Frass (droppings) on leaves"],
                "control_methods": ["Biological control (Trichogramma wasps)", "Spray Bacillus thuringiensis (Bt)",
                                    "Use pheromone traps", "Hand-picking early morning", "Neem oil spray"]
            },
            "rice_black_bug": {
                "scientific_name": "Scotinophara coarctata",
                "local_name": "Itim na Atangya / Black Bug",
                "crops_affected": ["rice"],
                "description": "Small black insects that suck sap from rice stems causing 'bugburn'. Found in rice terraces of Ifugao, Kalinga.",
                "visual_signs": ["Yellowing plants", "Stunted growth", "Black insects on stems", "Burnt appearance"],
                "control_methods": ["Light trapping during full moon", "Herding ducks in the field",
                                    "Submerge eggs by raising water level", "Spray neem extract"]
            },
            "brown_planthopper": {
                "scientific_name": "Nilaparvata lugens",
                "local_name": "Kayumangging Atangya",
                "crops_affected": ["rice"],
                "description": "Causes 'hopperburn' (browning and drying of crops). Transmits Ragged Stunt Virus.",
                "control_methods": ["Use resistant varieties (NSIC Rc)", "Avoid excessive nitrogen fertilizer",
                                    "Synchronous planting"]
            },
            "corn_borer": {
                "scientific_name": "Ostrinia furnacalis",
                "local_name": "Uod ng Mais",
                "crops_affected": ["corn"],
                "description": "Larvae bore into stalks and ears. The most destructive corn pest in PH.",
                "control_methods": ["Detasseling", "Trichogramma release", "Planting Bt Corn (if approved)"]
            },
            "cocolisap": {
                "scientific_name": "Aspidiotus rigidus",
                "local_name": "Cocolisap",
                "crops_affected": ["coconut", "lanzones"],
                "description": "Scale insects covering leaves, blocking photosynthesis. Historic outbreak in CALABARZON.",
                "control_methods": ["Pruning and burning affected parts", "Systemic trunk injection (FPA approved)",
                                    "Release of biocontrol agents"]
            },
            "mango_cecid_fly": {
                "scientific_name": "Procontarinia spp.",
                "local_name": "Kurikong",
                "crops_affected": ["mango"],
                "description": "Causes circular, brown, scab-like lesions on fruit skin.",
                "control_methods": ["Pruning overcrowded branches", "Bagging fruits early", "Proper orchard sanitation"]
            },
            "stem_borer": {
                "scientific_name": "Scirpophaga incertulas",
                "local_name": "Aksip / Stem Borer",
                "crops_affected": ["rice"],
                "description": "Larvae bore into stem causing 'deadheart' (young stage) or 'whitehead' (reproductive stage).",
                "control_methods": ["Light traps", "Pheromone traps", "Conservation of natural enemies"]
            }
        }

    def _build_disease_database(self) -> Dict:
        """Build Philippine agricultural disease database (Verified List)"""
        return {
            "rice_blast": {
                "scientific_name": "Magnaporthe oryzae",
                "local_name": "Leeg-leeg (Neck Blast)",
                "crops_affected": ["rice"],
                "description": "Diamond-shaped lesions on leaves or rotting of the panicle neck.",
                "control_methods": ["Avoid excessive nitrogen", "Keep field flooded",
                                    "Use fungicides (Tricyclazole) as last resort"]
            },
            "tungro": {
                "scientific_name": "Rice Tungro Bacilliform Virus",
                "local_name": "Tungro",
                "crops_affected": ["rice"],
                "description": "Viral disease causing yellow-orange discoloration of leaves and stunted growth. Vectored by Green Leafhopper. Found in rice terraces.",
                "visual_signs": ["Yellow to orange leaves", "Stunted plants", "Reduced tillering", "Mottled discoloration"],
                "control_methods": ["Plant resistant varieties (Matatag lines)", "Control leafhopper vectors",
                                    "Roguing (removal) of infected plants", "Use virus-free seeds"]
            },
            "bacterial_leaf_blight": {
                "scientific_name": "Xanthomonas oryzae",
                "local_name": "Kuyog / Leaf Blight",
                "crops_affected": ["rice"],
                "description": "Bacterial disease causing yellowing and drying of leaf tips and margins. Common in wet season rice.",
                "visual_signs": ["Yellow leaf tips", "V-shaped lesions", "Water-soaked areas", "Bacterial ooze", "Wilting"],
                "control_methods": ["Balanced fertilization", "Proper drainage", "Clean field sanitation", "Resistant varieties", "Reduce nitrogen"]
            },
            "late_blight": {
                "scientific_name": "Phytophthora infestans",
                "local_name": "Late Blight / Peste",
                "crops_affected": ["potato", "tomato"],
                "description": "Devastating fungal disease in cool, wet conditions. Major threat to CAR potato and tomato crops.",
                "visual_signs": ["Dark brown lesions", "White fuzzy growth", "Rapid plant collapse", "Rotting tubers", "Water-soaked spots"],
                "control_methods": ["Fungicide spray (Mancozeb, Ridomil)", "Destroy infected plants", "Improve air circulation", "Resistant varieties", "Avoid overhead irrigation"]
            },
            "downy_mildew": {
                "scientific_name": "Various Peronospora spp.",
                "local_name": "Downy Mildew / Amag",
                "crops_affected": ["lettuce", "cabbage", "broccoli", "spinach", "onion"],
                "description": "Fungal disease causing yellow spots on upper leaves with fuzzy growth underneath. Common in CAR highland vegetables.",
                "visual_signs": ["Yellow patches on leaves", "White/gray fuzzy growth underneath", "Curled leaves", "Stunted growth"],
                "control_methods": ["Copper fungicide", "Improve air circulation", "Avoid overhead watering", "Remove infected leaves", "Crop rotation"]
            },
            "powdery_mildew": {
                "scientific_name": "Various Erysiphales",
                "local_name": "Powdery Mildew / Pulbos",
                "crops_affected": ["squash", "cucumber", "beans", "peas", "strawberry"],
                "description": "White powdery coating on leaves and stems. Common in CAR during dry season.",
                "visual_signs": ["White powder on leaves", "Curled leaves", "Yellowing", "Stunted growth", "Reduced yield"],
                "control_methods": ["Sulfur spray", "Neem oil", "Baking soda solution", "Milk spray (1:10)", "Improve air flow", "Resistant varieties"]
            },
            "anthracnose": {
                "scientific_name": "Colletotrichum spp.",
                "local_name": "Anthracnose / Tagulubong",
                "crops_affected": ["mango", "strawberry", "beans", "pepper", "tomato"],
                "description": "Fungal disease causing dark sunken lesions on fruits and leaves. Problem in CAR fruit crops.",
                "visual_signs": ["Dark sunken spots", "Brown/black lesions", "Fruit rot", "Premature fruit drop"],
                "control_methods": ["Copper fungicide", "Remove infected parts", "Improve drainage", "Prune for air flow", "Post-harvest treatment"]
            },
            "mosaic_virus": {
                "scientific_name": "Various Potyvirus spp.",
                "local_name": "Mosaic Virus",
                "crops_affected": ["beans", "cucumber", "squash", "tomato", "pepper"],
                "description": "Viral disease causing mottled, mosaic-like patterns on leaves. Spread by aphids and contact.",
                "visual_signs": ["Mottled yellow-green patterns", "Distorted leaves", "Stunted growth", "Reduced yield"],
                "control_methods": ["Control aphid vectors", "Remove infected plants", "Use virus-free seeds", "Disinfect tools", "Plant resistant varieties"]
            }
        }

    async def analyze_image(self, image_data: bytes, filename: str, context: str = "") -> Dict:
        """
        Analyze image using Ollama Vision (LLaVA/Moondream) with ROBUST reasoning chain.
        """
        try:
            logger.info(f"Analyzing image {filename} using model: {self.ollama_vision_model}")

            # ENHANCED PROMPT: Multi-step analysis with CAR-specific context
            prompt = f"""
            You are an expert agricultural diagnostician specializing in the Cordillera Administrative Region (CAR) of the Philippines.
            CAR includes highland areas: Benguet, Ifugao, Kalinga, Abra, Apayao, Mountain Province - known for rice terraces, vegetable farming, and cool-climate crops.

            Context: {context}

            ANALYZE THIS IMAGE IN 4 STEPS:

            STEP 1: AGRICULTURAL VALIDATION
            - Examine the image carefully. Is this clearly a CROP, PLANT, or AGRICULTURAL PEST?
            - Agricultural = rice, corn, vegetables (cabbage, lettuce, tomato, potato, beans, pechay), fruits, or visible insects/diseases on plants
            - NOT Agricultural = people, selfies, buildings, vehicles, documents, random objects, unclear/blurry images
            - If NOT agricultural, set is_agricultural: false and STOP.

            STEP 2: PLANT IDENTIFICATION
            - Identify the specific crop or plant type
            - Common CAR crops: rice (terraced), corn, cabbage, lettuce, potato, tomato, carrots, beans, strawberry, pechay, sayote
            - If uncertain, use "Unknown Crop" or the plant family

            STEP 3: DETAILED VISUAL INSPECTION
            Look for these specific signs:
            PEST INDICATORS:
            - Holes or chewed leaves (armyworm, cabbage worm)
            - Visible insects: caterpillars, aphids (clusters), whiteflies (white dots), beetles
            - Frass (insect droppings) on leaves
            - Webbing or sticky residue
            - Curled or distorted leaves from feeding

            DISEASE INDICATORS:
            - Leaf spots: brown, black, yellow patches
            - Discoloration patterns: yellowing, browning, mosaic patterns
            - Fuzzy growth: white powder (powdery mildew), gray fuzz (downy mildew)
            - Wilting or drooping despite water
            - Rotting stems or fruits
            - V-shaped lesions, water-soaked areas

            HEALTHY SIGNS:
            - Green, uniform color
            - No visible damage or spots
            - Strong, upright growth

            STEP 4: DIAGNOSIS & RESPONSE
            - Identify the specific issue (e.g., "Armyworm", "Late Blight", "Aphid Infestation", "Powdery Mildew")
            - If healthy or minor issues, mark condition as "Healthy" or "Minor Stress"
            - Provide confidence (0-100): Use 80-100 for clear cases, 50-79 for probable, below 50 for uncertain
            - Write natural_response as a helpful explanation for a Filipino farmer in English

            OUTPUT VALID JSON ONLY (no other text):
            {{
                "is_agricultural": boolean,
                "plant_name": "string (e.g., 'Cabbage', 'Rice', 'Potato')",
                "detected_issue": "string (e.g., 'Armyworm', 'Late Blight', 'Aphids', 'None')",
                "condition": "Healthy" | "Pest Detected" | "Disease Detected" | "Minor Stress" | "N/A",
                "confidence_score": number (0-100),
                "natural_response": "string (helpful explanation)",
                "visual_details": "string (describe what you see: colors, patterns, damage type)"
            }}
            """

            response = ollama.chat(
                model=self.ollama_vision_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_data]
                }]
            )

            content = response['message']['content']
            logger.info(f"Ollama Raw Vision Response: {content}")

            return self._parse_llm_analysis(content)

        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return self._get_fallback_analysis()

    def _parse_llm_analysis(self, content: str) -> Dict:
        """Parse the LLM's text response into structured data"""
        try:
            # Robust JSON extraction (handles cases where LLM puts text before/after JSON)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

            # 1. STRICT REJECTION FOR NON-AGRI
            # If explicit false, or if confidence is very low
            is_agricultural = data.get("is_agricultural", False)
            confidence = data.get("confidence_score", 100)

            if not is_agricultural or confidence < 40:
                return {
                    "plant_type": "Non-Agricultural Object",
                    "pest_detected": False,
                    "disease_detected": False,
                    "health_status": "Not Agricultural",
                    "severity": "None",
                    "recommendations": ["Please upload a clear photo of a crop, plant, or pest."],
                    "sources": [],
                    "natural_summary": data.get("natural_response",
                                                "I'm sorry, I couldn't recognize a plant or crop in this image.")
                }

            # 2. Process Agricultural Data
            plant_type = data.get("plant_name", "Unknown Crop")
            detected_issue = data.get("detected_issue", "")
            condition = data.get("condition", "Unknown")
            natural_summary = data.get("natural_response", "")
            visual_details = data.get("visual_details", "")
            confidence = data.get("confidence_score", 50)

            # Add visual details to summary if available
            if visual_details and len(natural_summary) < 100:
                natural_summary += f" Visual observation: {visual_details}"

            pest_detected = "PEST" in condition.upper()
            disease_detected = "DISEASE" in condition.upper()
            
            # Enhance severity based on confidence and condition
            if confidence >= 80 and (pest_detected or disease_detected):
                severity = "High"
            elif confidence >= 60 and (pest_detected or disease_detected):
                severity = "Moderate"
            elif pest_detected or disease_detected:
                severity = "Low to Moderate"
            else:
                severity = "None"

            # Enhanced Database Matching for Recommendations
            recommendations = []
            pest_info = None
            disease_info = None

            content_upper = str(detected_issue).upper()
            plant_upper = str(plant_type).upper()

            # Match specific issue to Verified Database (improved fuzzy matching)
            if pest_detected:
                for key, info in self.pest_database.items():
                    # Check key, local names (split by /), and common words
                    key_parts = key.replace("_", " ").upper().split()
                    local_parts = info['local_name'].upper().split("/")
                    
                    # Check if any part matches
                    if (any(part in content_upper for part in key_parts) or
                        any(part.strip() in content_upper for part in local_parts) or
                        key.replace("_", "") in content_upper.replace(" ", "")):
                        
                        # Verify crop match if possible
                        if plant_upper != "UNKNOWN CROP":
                            crop_match = any(crop.upper() in plant_upper or plant_upper in crop.upper() 
                                           for crop in info['crops_affected'])
                            if not crop_match:
                                continue  # Skip if crop doesn't match
                        
                        recommendations = info['control_methods'].copy()
                        pest_info = info
                        if len(natural_summary) < 80:
                            natural_summary += f" This appears to be {info['local_name']}, affecting {', '.join(info['crops_affected'][:3])}."
                        break

            if disease_detected:
                for key, info in self.disease_database.items():
                    key_parts = key.replace("_", " ").upper().split()
                    local_parts = info['local_name'].upper().split("/")
                    
                    if (any(part in content_upper for part in key_parts) or
                        any(part.strip() in content_upper for part in local_parts) or
                        key.replace("_", "") in content_upper.replace(" ", "")):
                        
                        if plant_upper != "UNKNOWN CROP":
                            crop_match = any(crop.upper() in plant_upper or plant_upper in crop.upper()
                                           for crop in info['crops_affected'])
                            if not crop_match:
                                continue
                        
                        recommendations = info['control_methods'].copy()
                        disease_info = info
                        if len(natural_summary) < 80:
                            natural_summary += f" This appears to be {info['local_name']}, common in {', '.join(info['crops_affected'][:3])}."
                        break

            # Fallback recommendations
            if not recommendations:
                if pest_detected:
                    recommendations = [
                        "Identify the specific pest for targeted control",
                        "Try neem oil spray as organic option",
                        "Remove heavily infested parts",
                        "Consult CAR Agricultural Extension Office"
                    ]
                elif disease_detected:
                    recommendations = [
                        "Improve air circulation around plants",
                        "Remove and destroy infected plant parts",
                        "Apply appropriate fungicide or bactericide",
                        "Practice crop rotation",
                        "Contact your local DA-CAR technician"
                    ]
                else:
                    recommendations = [
                        "Continue Good Agricultural Practices (GAP)",
                        "Monitor regularly for early pest/disease detection",
                        "Maintain proper spacing and ventilation",
                        "Keep records of crop health"
                    ]
                    if "healthy" in condition.lower() and len(natural_summary) < 20:
                        natural_summary = f"The {plant_type} looks healthy! Continue your good farming practices."

            return {
                "plant_type": plant_type,
                "pest_detected": pest_detected,
                "disease_detected": disease_detected,
                "health_status": condition,
                "severity": severity,
                "pest_info": pest_info,
                "disease_info": disease_info,
                "recommendations": recommendations,
                "sources": [],
                "natural_summary": natural_summary,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"JSON Parse error: {str(e)}")
            # Fallback to text parsing if JSON fails but might still be valid text
            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict:
        """Strict fallback that admits failure"""
        return {
            "plant_type": "Analysis Failed",
            "pest_detected": False,
            "disease_detected": False,
            "health_status": "System Error",
            "natural_summary": "I'm sorry, I couldn't analyze that image clearly. Please try again with a better photo.",
            "recommendations": [],
            "sources": []
        }


# Initialize service
image_analysis_service = ImageAnalysisService()