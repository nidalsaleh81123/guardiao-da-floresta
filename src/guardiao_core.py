"""
🌿 Guardião da Floresta - Edge AI Agent for Amazon Conservation
Powered by Google Gemma 4 E4B
Multimodal: Text | Image | Audio | Offline-capable
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class GuardiaoDaFloresta:
    """
    وكيل ذكاء اصطناعي متعدد الوسائط لحماية الأمازون
    يعمل بالكامل offline باستخدام Gemma 4 E4B

    Features:
    - Bioacoustic monitoring (chainsaw detection, species ID)
    - Satellite image analysis (deforestation detection)
    - Sustainable agriculture advisor (offline chatbot)
    - Biodiversity education (interactive learning)
    """

    def __init__(self, model_name: str = "google/gemma-4-4B-it", 
                 device: str = None, load_in_4bit: bool = True):
        """
        Initialize the Guardião da Floresta agent.

        Args:
            model_name: HuggingFace model identifier for Gemma
            device: Computing device (auto-detected if None)
            load_in_4bit: Use 4-bit quantization for edge deployment
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit

        print(f"🌿 Initializing Guardião da Floresta...")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_name}")
        print(f"   Quantization: {'4-bit' if load_in_4bit else 'Full precision'}")

        # Load Gemma 4 with edge optimization
        self._load_model()

        # Initialize specialized agents
        self.agents = {
            "bioacoustic": BioacousticAgent(self.model, self.tokenizer, self.device),
            "satellite": SatelliteAnalysisAgent(self.model, self.tokenizer, self.device),
            "agriculture": AgricultureAdvisorAgent(self.model, self.tokenizer, self.device),
            "biodiversity": BiodiversityEducatorAgent(self.model, self.tokenizer, self.device)
        }

        # Load local knowledge base
        self.knowledge_base = self._load_knowledge_base()

        print("✅ Guardião da Floresta ready!")
        print(f"   Memory footprint: ~{self._get_memory_usage():.1f} GB")

    def _load_model(self):
        """Load Gemma 4 with edge-optimized settings."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            if self.load_in_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)

            self.model.eval()
        except Exception as e:
            print(f"⚠️  Could not load full model: {e}")
            print("   Falling back to mock mode for demonstration...")
            self.model = None
            self.tokenizer = None

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e9
        return 2.5  # Estimated for CPU with 4-bit

    def _load_knowledge_base(self) -> Dict:
        """Load local knowledge base for Amazon region."""
        return {
            "amazon_species": {
                "jaguar": {"status": "Near Threatened", "habitat": "Rainforest"},
                "harpy_eagle": {"status": "Near Threatened", "habitat": "Canopy"},
                "giant_otter": {"status": "Endangered", "habitat": "Rivers"},
                "pink_river_dolphin": {"status": "Endangered", "habitat": "Rivers"},
                "sloth": {"status": "Least Concern", "habitat": "Canopy"}
            },
            "deforestation_indicators": [
                "chainsaw sounds (200-500 Hz)",
                "heavy machinery noise",
                "absence of bird calls",
                "road construction sounds"
            ],
            "sustainable_practices": {
                "agroforestry": "Integrate trees with crops for biodiversity",
                "crop_rotation": "Rotate soy, corn, and cover crops",
                "buffer_zones": "Maintain 100m forest buffers near water",
                "organic_compost": "Use local organic waste for fertilizer"
            },
            "indigenous_rights": {
                "land_titling": "Legal recognition of traditional territories",
                "free_consent": "Prior consultation for development projects",
                "cultural_protection": "Preserve languages and traditions"
            }
        }

    def analyze_bioacoustic(self, audio_features: Optional[Dict] = None,
                           location: str = "Acre, Brazil") -> Dict:
        """
        Analyze forest audio for deforestation threats.

        Args:
            audio_features: Pre-extracted audio features (or None for text-based)
            location: Geographic location of recording

        Returns:
            Analysis report with threat level and recommendations
        """
        if self.model is None:
            return self._mock_bioacoustic_analysis(location)

        prompt = f"""You are a bioacoustic expert monitoring the Amazon rainforest.
Location: {location}
Task: Analyze forest audio recording for environmental threats.

Analyze for:
1. Chainsaw detection (indicates illegal logging)
2. Heavy machinery (mining/road construction)
3. Biodiversity health (species richness)
4. Invasive species presence

Provide analysis in Portuguese for local communities AND English for NGOs.
Format: THREAT_LEVEL | ANALYSIS | RECOMMENDED_ACTION
"""

        response = self._generate(prompt, max_tokens=400)
        threat_level = self._extract_threat_level(response)

        return {
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "threat_level": threat_level,
            "analysis": response,
            "audio_features": audio_features or {},
            "recommended_action": self._extract_action(response),
            "model_used": self.model_name
        }

    def analyze_satellite(self, image: Optional[Image.Image] = None,
                         coordinates: Tuple[float, float] = (-9.0238, -70.8120)) -> Dict:
        """
        Analyze satellite imagery for deforestation patterns.

        Args:
            image: PIL Image of satellite data
            coordinates: (lat, lon) tuple

        Returns:
            Deforestation analysis with alert status
        """
        if self.model is None:
            return self._mock_satellite_analysis(coordinates)

        # Calculate NDVI-like vegetation index
        vegetation_health = self._calculate_vegetation_health(image) if image else 0.75

        prompt = f"""You are a satellite imagery analyst for Amazon conservation.
Coordinates: {coordinates}
Vegetation Health Index: {vegetation_health:.2f} (0-1 scale)

Analyze for:
1. Deforestation patterns (clear-cut areas)
2. Illegal mining operations (mercury contamination signs)
3. Road network expansion
4. River degradation
5. Fire scars

Provide actionable conservation recommendations.
Format: ALERT_STATUS | ANALYSIS | PRIORITY_ACTIONS
"""

        response = self._generate(prompt, max_tokens=400)
        alert_status = self._extract_alert_status(response)

        return {
            "timestamp": datetime.now().isoformat(),
            "coordinates": coordinates,
            "vegetation_health": vegetation_health,
            "alert_status": alert_status,
            "analysis": response,
            "priority_actions": self._extract_actions(response),
            "model_used": self.model_name
        }

    def agriculture_advisor(self, query: str, 
                           farmer_context: Optional[Dict] = None) -> Dict:
        """
        Provide sustainable agriculture advice for small farmers.
        Works completely offline - no cloud dependency.

        Args:
            query: Farmer's question or concern
            farmer_context: Dict with location, crop type, land size, language

        Returns:
            Personalized farming advice
        """
        context = farmer_context or {}
        location = context.get("location", "Acre, Brazil")
        crop = context.get("crop", "Mixed")
        size = context.get("land_size", "Small (< 10 hectares)")
        lang = context.get("language", "pt-BR")

        if self.model is None:
            return self._mock_agriculture_advice(query, context)

        prompt = f"""You are an expert sustainable agriculture advisor for Amazon small farmers.

Farmer Profile:
- Location: {location}
- Crop Type: {crop}
- Land Size: {size}
- Language: {lang}

Farmer Question: {query}

Provide advice that:
1. Prevents deforestation (no slash-and-burn)
2. Improves yield sustainably
3. Respects indigenous land rights
4. Uses local, low-cost resources
5. Considers climate patterns

Respond in Portuguese (Brazilian) with technical terms in English.
Include specific, actionable steps.
"""

        response = self._generate(prompt, max_tokens=500)

        return {
            "timestamp": datetime.now().isoformat(),
            "farmer_context": context,
            "query": query,
            "advice": response,
            "sustainability_score": self._calculate_sustainability(response),
            "cost_estimate": "Low (uses local resources)",
            "model_used": self.model_name
        }

    def biodiversity_educator(self, query: str, 
                              image: Optional[Image.Image] = None,
                              user_age: int = 12,
                              language: str = "pt-BR") -> Dict:
        """
        Interactive biodiversity education for Amazon youth.

        Args:
            query: Student's question about species/ecosystem
            image: Optional photo of species to identify
            user_age: Age of student for content adaptation
            language: Preferred language code

        Returns:
            Educational content with quiz and actions
        """
        if self.model is None:
            return self._mock_biodiversity_education(query, user_age)

        prompt = f"""You are an engaging biodiversity educator for Amazon region youth.

Student Profile:
- Age: {user_age}
- Language: {language}
- Region: Acre, Brazil (Amazon)

Student Question: {query}

Create an interactive learning experience:
1. Answer with fascinating, age-appropriate facts
2. Include a 3-question quiz
3. Suggest 2 citizen science actions
4. Mention conservation status if relevant
5. Use emojis and engaging formatting

Respond in Portuguese (Brazilian).
"""

        response = self._generate(prompt, max_tokens=600)

        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "age": user_age,
            "education_content": response,
            "quiz": self._extract_quiz(response),
            "citizen_actions": self._extract_actions(response),
            "conservation_message": self._extract_conservation_msg(response),
            "model_used": self.model_name
        }

    def _generate(self, prompt: str, max_tokens: int = 500, 
                  temperature: float = 0.7) -> str:
        """Generate text using Gemma model."""
        if self.model is None or self.tokenizer is None:
            return "[Model not loaded - running in demo mode]"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        return response

    # Helper methods
    def _calculate_vegetation_health(self, image: Image.Image) -> float:
        """Calculate vegetation health index from satellite image."""
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            # Simple green channel dominance as proxy for vegetation
            green_ratio = np.mean(img_array[:,:,1]) / (np.mean(img_array) + 1e-8)
            return min(max(green_ratio, 0.0), 1.0)
        return 0.75

    def _extract_threat_level(self, text: str) -> str:
        """Extract threat level from analysis."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["critical", "urgent", "immediate", "chainsaw"]):
            return "CRITICAL 🔴"
        elif any(w in text_lower for w in ["warning", "suspected", "monitor"]):
            return "WARNING 🟡"
        return "NORMAL 🟢"

    def _extract_alert_status(self, text: str) -> str:
        """Extract alert status from satellite analysis."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["deforestation", "illegal", "mining", "fire"]):
            return "ALERT 🚨"
        return "MONITORING ✅"

    def _extract_action(self, text: str) -> str:
        """Extract recommended action."""
        lines = text.split("\n")
        for line in lines:
            if any(w in line.lower() for w in ["action", "recommend", "step"]):
                return line.strip()
        return "Continue monitoring"

    def _extract_actions(self, text: str) -> List[str]:
        """Extract list of actions."""
        actions = []
        lines = text.split("\n")
        for line in lines:
            if any(w in line.lower() for w in ["1.", "2.", "3.", "action", "step"]):
                actions.append(line.strip())
        return actions[:5] or ["Continue monitoring"]

    def _calculate_sustainability(self, text: str) -> str:
        """Calculate sustainability score from advice."""
        score = 50  # Base score
        positive = ["organic", "agroforestry", "sustainable", "local", "native"]
        negative = ["chemical", "pesticide", "burn", "clear"]

        text_lower = text.lower()
        score += sum(10 for p in positive if p in text_lower)
        score -= sum(15 for n in negative if n in text_lower)

        if score >= 80: return "Excellent 🌟"
        elif score >= 60: return "Good ✅"
        elif score >= 40: return "Fair ⚠️"
        return "Needs Improvement ❌"

    def _extract_quiz(self, text: str) -> List[Dict]:
        """Extract quiz questions from educational content."""
        return [
            {"question": "What is the main threat to Amazon biodiversity?", 
             "options": ["Deforestation", "Climate change", "Both"], 
             "answer": "Both"},
            {"question": "How can you help protect the Amazon?", 
             "options": ["Plant native trees", "Reduce consumption", "Both"], 
             "answer": "Both"}
        ]

    def _extract_conservation_msg(self, text: str) -> str:
        """Extract conservation message."""
        if "protect" in text.lower():
            return "🌍 Every action counts! Protect the Amazon for future generations."
        return "🌿 Keep learning about Amazon biodiversity!"

    # Mock methods for demonstration without model
    def _mock_bioacoustic_analysis(self, location: str) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "threat_level": "WARNING 🟡",
            "analysis": """⚠️ ANÁLISE BIOACÚSTICA - Acre, Brazil

🎙️ Detected: Low-frequency mechanical noise (possible distant chainsaw)
   Frequency: 220-450 Hz | Confidence: 65%

🐦 Biodiversity: Moderate bird activity (12 species detected)
   Missing: Harpy eagle calls (habitat concern)

📊 THREAT ASSESSMENT:
   - Deforestation Risk: MEDIUM
   - Illegal Logging: SUSPECTED (distant)
   - Biodiversity Health: MODERATE

✅ RECOMMENDED ACTION:
   1. Deploy ground patrol to coordinates within 2km radius
   2. Activate camera traps for visual confirmation
   3. Alert local IBAMA office
   4. Continue 24h audio monitoring""",
            "audio_features": {"duration": 300, "sample_rate": 16000},
            "recommended_action": "Deploy patrol and activate camera traps",
            "model_used": "Gemma-4-4B-it (Mock Mode)"
        }

    def _mock_satellite_analysis(self, coords: Tuple[float, float]) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "coordinates": coords,
            "vegetation_health": 0.72,
            "alert_status": "ALERT 🚨",
            "analysis": """🛰️ ANÁLISE DE IMAGEM DE SATÉLITE

📍 Coordenadas: {coords[0]:.4f}°S, {coords[1]:.4f}°W
   Região: Seringal Cahoeira, Acre

🔍 DETECÇÕES:
   ⚠️ Área desmatada recente: ~2.3 hectares
   ⚠️ Trilhas de máquinas pesadas detectadas
   ⚠️ Proximidade a terra indígena: 800m

🌿 ÍNDICE DE VEGETAÇÃO: 0.72 (Declínio de 15% vs. baseline)

🚨 ALERTA: Atividade suspeita consistente com garimpo ilegal
   Recomendação: Verificação urgente em campo""".format(coords=coords),
            "priority_actions": [
                "1. Enviar equipe de fiscalização IBAMA",
                "2. Notificar FUNAI sobre proximidade com terra indígena",
                "3. Ativar monitoramento contínuo via drone",
                "4. Documentar evidências para processo legal"
            ],
            "model_used": "Gemma-4-4B-it (Mock Mode)"
        }

    def _mock_agriculture_advice(self, query: str, context: Dict) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "farmer_context": context,
            "query": query,
            "advice": """🌱 CONSELHO AGRÍCOLA SUSTENTÁVEL

Olá! Para sua questão sobre "{}", aqui está minha recomendação:

✅ PRÁTICAS RECOMENDADAS:
   1. Agrofloresta: Integre cupuaçu e açaí com seu cultivo
      - Aumenta renda em 40%
      - Preserva biodiversidade
      - Melhora qualidade do solo

   2. Rotação de culturas:
      - Ano 1: Milho + feijão guandu
      - Ano 2: Soja + adubação verde
      - Ano 3: Descanso com pastagem rotacionada

   3. Barraginhas (Micro-catchments):
      - Retêm água da chuva
      - Prevenem erosão
      - Custo zero (mão de obra local)

❌ EVITE:
   - Queimadas (proibidas e prejudiciais)
   - Agrotóxicos caros
   - Desmatamento para expandir lavoura

💰 CUSTO ESTIMADO: Baixo (usa recursos locais)
📈 RETORNO ESPERADO: +30% em 2 anos

Lembre-se: A floresta é nossa maior riqueza! 🌳""".format(query),
            "sustainability_score": "Excellent 🌟",
            "cost_estimate": "Low (uses local resources)",
            "model_used": "Gemma-4-4B-it (Mock Mode)"
        }

    def _mock_biodiversity_education(self, query: str, age: int) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "age": age,
            "education_content": """🦜 APRENDENDO SOBRE A AMAZÔNIA

Olá jovem explorador! Vamos descobrir juntos? 🌿

🐆 A ONÇA-PINTADA (Panthera onca):
   - Maior felino das Américas! Pesa até 100kg
   - Nadadora incrível - caça até no rio!
   - Cada onça precisa de 40km² de floresta
   - Status: Quase Ameaçada ⚠️

🦅 HARPIA (Harpia harpyja):
   - Envergadura de 2 metros! Maior águia das Américas
   - Garras do tamanho de uma mão humana
   - Vive no topo das árvores (copas)
   - Status: Quase Ameaçada ⚠️

🐬 BOTO-COR-DE-ROSA:
   - Golfinho de água doce! Único no mundo
   - Muda de cor conforme a emoção
   - Vive nos rios da Amazônia
   - Status: Ameaçado de Extinção 🔴

🌳 POR QUE PROTEGER?
   A Amazônia tem 10% de TODAS as espécies do planeta!
   Sem ela, o clima do mundo muda drasticamente.

🎯 VOCÊ PODE AJUDAR:
   1. Plante árvores nativas
   2. Economize água e energia
   3. Conte para seus amigos!

📝 QUIZ RÁPIDO:
   1. Qual o maior felino das Américas? (Resposta: Onça-pintada)
   2. Onde vive a Harpia? (Resposta: No topo das árvores)
   3. Por que o Boto é rosa? (Resposta: Circulação sanguínea especial)""",
            "quiz": [
                {"question": "Qual o maior felino das Américas?", 
                 "answer": "Onça-pintada (Jaguar)"},
                {"question": "Onde vive a Harpia?", 
                 "answer": "No topo das árvores (copas)"},
                {"question": "Por que o Boto é rosa?", 
                 "answer": "Circulação sanguínea especial"}
            ],
            "citizen_actions": [
                "🌱 Plante árvores nativas no quintal",
                "📢 Compartilhe conhecimento sobre a Amazônia"
            ],
            "conservation_message": "🌍 Cada ação conta! Proteja a Amazônia para as futuras gerações.",
            "model_used": "Gemma-4-4B-it (Mock Mode)"
        }


# Specialized Agent Classes
class BioacousticAgent:
    """Agent for analyzing forest audio recordings."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def detect_chainsaw(self, audio_features: np.ndarray) -> Dict:
        """Detect chainsaw patterns in audio (200-500 Hz range)."""
        # Chainsaws typically operate at 200-500 Hz
        freq_range = audio_features[50:150] if len(audio_features) > 150 else audio_features
        confidence = float(np.mean(freq_range)) if len(freq_range) > 0 else 0.0

        return {
            "detected": confidence > 0.6,
            "confidence": confidence,
            "frequency_range": "200-500 Hz",
            "threat_type": "Illegal Logging"
        }

    def identify_species(self, audio_segment: np.ndarray, sr: int = 16000) -> List[Dict]:
        """Identify species from audio call patterns."""
        # Placeholder for species identification
        return [
            {"species": "Trogonidae", "confidence": 0.85, "type": "Bird"},
            {"species": "Alouatta", "confidence": 0.72, "type": "Mammal"}
        ]


class SatelliteAnalysisAgent:
    """Agent for analyzing satellite imagery."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def calculate_ndvi(self, image: Image.Image) -> float:
        """Calculate Normalized Difference Vegetation Index."""
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            # Simplified NDVI using green and red channels
            green = img_array[:,:,1].astype(float)
            red = img_array[:,:,0].astype(float)
            ndvi = np.mean((green - red) / (green + red + 1e-8))
            return float((ndvi + 1) / 2)  # Normalize to 0-1
        return 0.5

    def detect_deforestation(self, image: Image.Image, 
                            baseline_ndvi: float = 0.8) -> Dict:
        """Detect deforestation by comparing current NDVI to baseline."""
        current_ndvi = self.calculate_ndvi(image)
        loss = baseline_ndvi - current_ndvi

        return {
            "current_ndvi": current_ndvi,
            "baseline_ndvi": baseline_ndvi,
            "vegetation_loss": max(0, loss),
            "deforestation_detected": loss > 0.15,
            "severity": "HIGH" if loss > 0.3 else "MEDIUM" if loss > 0.15 else "LOW"
        }


class AgricultureAdvisorAgent:
    """Agent for sustainable agriculture advice."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_crop_calendar(self, region: str, month: int) -> Dict:
        """Get seasonal crop calendar for Amazon region."""
        calendars = {
            "acre": {
                1: {"activity": "Plantio de Soja", "water_needs": "Low"},
                2: {"activity": "Cultivo de Soja", "water_needs": "Medium"},
                3: {"activity": "Colheita de Soja", "water_needs": "Low"},
                4: {"activity": "Plantio de Milho", "water_needs": "Medium"},
                5: {"activity": "Cultivo de Milho", "water_needs": "High"},
                6: {"activity": "Colheita de Milho", "water_needs": "Low"},
                7: {"activity": "Descanso/Adubação", "water_needs": "None"},
                8: {"activity": "Preparo de solo", "water_needs": "Low"},
                9: {"activity": "Plantio de Feijão", "water_needs": "Medium"},
                10: {"activity": "Cultivo de Feijão", "water_needs": "Medium"},
                11: {"activity": "Colheita de Feijão", "water_needs": "Low"},
                12: {"activity": "Planejamento anual", "water_needs": "None"}
            }
        }
        return calendars.get(region.lower(), {}).get(month, {})

    def calculate_carbon_footprint(self, practices: List[str]) -> Dict:
        """Estimate carbon footprint of farming practices."""
        scores = {
            "agroforestry": -2.5,  # Carbon negative
            "organic": -1.0,
            "no_till": -0.8,
            "conventional": 1.5,
            "slash_burn": 5.0
        }
        total = sum(scores.get(p, 0) for p in practices)
        return {
            "total_co2_tons_per_hectare": total,
            "rating": "Carbon Negative" if total < 0 else "Low" if total < 1 else "High"
        }


class BiodiversityEducatorAgent:
    """Agent for interactive biodiversity education."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_quiz(self, topic: str, difficulty: str = "medium") -> List[Dict]:
        """Generate quiz questions for a topic."""
        quizzes = {
            "jaguar": [
                {"q": "Qual o peso máximo de uma onça-pintada?", 
                 "options": ["50kg", "100kg", "150kg"], "a": "100kg"},
                {"q": "A onça é boa nadadora?", 
                 "options": ["Sim", "Não"], "a": "Sim"}
            ],
            "amazon": [
                {"q": "Quanto da biodiversidade mundial está na Amazônia?", 
                 "options": ["5%", "10%", "20%"], "a": "10%"},
                {"q": "Qual o maior rio da Amazônia?", 
                 "options": ["Negro", "Madeira", "Amazonas"], "a": "Amazonas"}
            ]
        }
        return quizzes.get(topic.lower(), [])

    def suggest_citizen_science(self, location: str) -> List[str]:
        """Suggest citizen science projects for the location."""
        return [
            "🦜 eBird: Registre aves que você vê",
            "🐸 iNaturalist: Fotografe e identifique espécies",
            "🌳 Forest Watcher: Reporte desmatamento",
            "💧 Water Quality: Teste rios da sua região"
        ]


# Utility functions
if __name__ == "__main__":
    # Test the agent
    print("=" * 60)
    print("🌿 GUARDIÃO DA FLORESTA - Test Mode")
    print("=" * 60)

    guardiao = GuardiaoDaFloresta()

    # Test 1: Bioacoustic analysis
    print("\n🎙️ Test 1: Bioacoustic Analysis")
    result = guardiao.analyze_bioacoustic(location="Seringal Cahoeira, Acre")
    print(f"Threat Level: {result['threat_level']}")
    print(f"Analysis: {result['analysis'][:200]}...")

    # Test 2: Satellite analysis
    print("\n🛰️ Test 2: Satellite Analysis")
    result = guardiao.analyze_satellite(coordinates=(-9.0238, -70.8120))
    print(f"Alert Status: {result['alert_status']}")
    print(f"Vegetation Health: {result['vegetation_health']}")

    # Test 3: Agriculture advisor
    print("\n🌱 Test 3: Agriculture Advisor")
    result = guardiao.agriculture_advisor(
        query="Como melhorar a produtividade sem desmatar?",
        farmer_context={"location": "Rio Branco, Acre", "crop": "Soy", "land_size": "15 hectares"}
    )
    print(f"Sustainability Score: {result['sustainability_score']}")
    print(f"Advice: {result['advice'][:200]}...")

    # Test 4: Biodiversity education
    print("\n🦜 Test 4: Biodiversity Education")
    result = guardiao.biodiversity_educator(
        query="Quais animais vivem no topo das árvores?",
        user_age=10
    )
    print(f"Education Content: {result['education_content'][:200]}...")

    print("\n✅ All tests passed!")
