"""
Guardiao da Floresta - Core AI Agent
Multimodal Edge AI for Amazon Conservation
Powered by Google Gemma 4 E4B

Author: [Your Name]
Competition: Build with Gemma: Amazon Eco-Hack
Track: Main Track - Best Amazon Eco-Hack
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration & Constants
# ============================================================================

@dataclass
class GuardiaoConfig:
    """Configuration for Guardiao da Floresta"""
    model_name: str = "google/gemma-4-E4B-it"
    device: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    load_in_4bit: bool = True

    # Audio processing
    sample_rate: int = 16000
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512

    # Image processing
    image_size: int = 896

    # Alert thresholds
    chainsaw_freq_range: Tuple[float, float] = (200, 500)
    threat_critical_threshold: int = 3
    threat_warning_threshold: int = 1

# ============================================================================
# Threat Classification System
# ============================================================================

class ThreatClassifier:
    """Classifies environmental threats from multimodal inputs"""

    CRITICAL_KEYWORDS = [
        "chainsaw", "heavy machinery", "bulldozer", "illegal logging",
        "deforestation", "fire", "burning", "mining", "garimpo",
        "urgent", "immediate danger", "critical"
    ]

    WARNING_KEYWORDS = [
        "unusual", "suspicious", "change detected", "anomaly",
        "possible", "potential", "monitor closely"
    ]

    @classmethod
    def classify_text(cls, text: str) -> Tuple[str, int, List[str]]:
        """
        Classify threat level from text analysis
        Returns: (level, score, matched_keywords)
        """
        text_lower = text.lower()
        critical_matches = [kw for kw in cls.CRITICAL_KEYWORDS if kw in text_lower]
        warning_matches = [kw for kw in cls.WARNING_KEYWORDS if kw in text_lower]

        score = len(critical_matches) * 2 + len(warning_matches)

        if score >= 6 or len(critical_matches) >= 3:
            return "CRITICAL", score, critical_matches
        elif score >= 2 or len(critical_matches) >= 1:
            return "WARNING", score, critical_matches + warning_matches
        return "NORMAL", score, []

# ============================================================================
# Knowledge Base - Local Amazon Data
# ============================================================================

class AmazonKnowledgeBase:
    """Local knowledge base for Amazon-specific information"""

    def __init__(self):
        self.species_db = self._load_species()
        self.climate_data = self._load_climate()
        self.agriculture_guide = self._load_agriculture()
        self.indigenous_rights = self._load_rights()
        self.conservation_areas = self._load_conservation_areas()

    def _load_species(self) -> Dict:
        return {
            "birds": {
                "harpy_eagle": {"status": "Near Threatened", "habitat": "Canopy"},
                "scarlet_macaw": {"status": "Least Concern", "habitat": "Forest Edge"},
                "toucan": {"status": "Least Concern", "habitat": "Canopy"},
                "hoatzin": {"status": "Least Concern", "habitat": "Riverside"},
            },
            "mammals": {
                "jaguar": {"status": "Near Threatened", "habitat": "Dense Forest"},
                "giant_otter": {"status": "Endangered", "habitat": "Rivers"},
                "sloth": {"status": "Least Concern", "habitat": "Canopy"},
                "tapir": {"status": "Vulnerable", "habitat": "Forest Floor"},
            },
            "amphibians": {
                "poison_dart_frog": {"status": "Vulnerable", "habitat": "Forest Floor"},
                "glass_frog": {"status": "Least Concern", "habitat": "Near Streams"},
            }
        }

    def _load_climate(self) -> Dict:
        return {
            "acre": {
                "rainy_season": "November to March",
                "dry_season": "June to September",
                "avg_temp": "24-26C",
                "annual_rainfall": "1800-2200mm",
                "climate_type": "Tropical Monsoon"
            },
            "amazonas": {
                "rainy_season": "December to May",
                "dry_season": "June to November",
                "avg_temp": "25-28C",
                "annual_rainfall": "2000-3000mm",
                "climate_type": "Equatorial"
            }
        }

    def _load_agriculture(self) -> Dict:
        return {
            "sustainable_practices": [
                "Agroforestry systems (sistema agroflorestal)",
                "Crop rotation with legumes",
                "Organic composting (compostagem)",
                "Integrated pest management",
                "Water conservation techniques",
                "No-burn agriculture (agricultura sem queima)"
            ],
            "crops": {
                "acai": {"season": "Year-round", "water_needs": "High", "shade": "Partial"},
                "cocoa": {"season": "Year-round", "water_needs": "Medium", "shade": "Full"},
                "rubber": {"season": "Year-round", "water_needs": "Medium", "shade": "Partial"},
                "cassava": {"season": "8-12 months", "water_needs": "Low", "shade": "Full Sun"},
                "corn": {"season": "3-4 months", "water_needs": "Medium", "shade": "Full Sun"},
                "soy": {"season": "4-5 months", "water_needs": "Medium", "shade": "Full Sun"}
            },
            "soil_conservation": [
                "Cover crops (culturas de cobertura)",
                "Terracing in sloped areas",
                "Contour planting (plantio em nivel)",
                "Mulching with local materials"
            ]
        }

    def _load_rights(self) -> Dict:
        return {
            "indigenous_territories": [
                "Right to traditional lands (Terra Indigena)",
                "Free, prior, and informed consent (FPIC)",
                "Right to maintain traditional practices",
                "Protection from illegal invasions"
            ],
            "environmental_laws": [
                "Brazilian Forest Code (Codigo Florestal)",
                "Legal Reserve requirement: 80% in Amazon",
                "APP protection (Area de Preservacao Permanente)",
                "Environmental licensing for rural properties"
            ],
            "reporting_channels": [
                "IBAMA: 0800-61-8080",
                "ICMBio: (61) 2028-9300",
                "Federal Police environmental crimes",
                "Local environmental secretariats"
            ]
        }

    def _load_conservation_areas(self) -> Dict:
        return {
            "chico_mendes": {
                "type": "Extractive Reserve",
                "location": "Acre",
                "area_km2": 9315,
                "established": 1990
            },
            "serra_do_divisor": {
                "type": "National Park",
                "location": "Acre",
                "area_km2": 8468,
                "established": 1989
            },
            "mamiraua": {
                "type": "Sustainable Development Reserve",
                "location": "Amazonas",
                "area_km2": 11240,
                "established": 1996
            }
        }

    def get_species_info(self, species_name: str) -> Optional[Dict]:
        """Get information about a species"""
        for category, species in self.species_db.items():
            if species_name.lower() in species:
                return {
                    "name": species_name,
                    "category": category,
                    **species[species_name.lower()]
                }
        return None

    def get_climate_for_region(self, region: str) -> Dict:
        """Get climate data for a region"""
        return self.climate_data.get(region.lower(), {})

    def get_crop_calendar(self, crop: str, region: str) -> Dict:
        """Get crop calendar information"""
        crop_info = self.agriculture_guide["crops"].get(crop.lower(), {})
        climate = self.get_climate_for_region(region)
        return {**crop_info, "climate": climate}

# ============================================================================
# Main Guardiao Agent
# ============================================================================

class GuardiaoDaFloresta:
    """
    Guardiao da Floresta - Multimodal AI Agent

    A comprehensive edge-deployable AI system for Amazon conservation that combines:
    - Bioacoustic monitoring for deforestation detection
    - Satellite image analysis for vegetation monitoring
    - Sustainable agriculture advisory for local farmers
    - Biodiversity education for youth empowerment

    All powered by Google Gemma 4 E4B running locally.
    """

    def __init__(self, config: Optional[GuardiaoConfig] = None):
        self.config = config or GuardiaoConfig()
        self.knowledge_base = AmazonKnowledgeBase()
        self.threat_classifier = ThreatClassifier()

        print("Initializing Guardiao da Floresta...")
        print("   Modelo: Google Gemma 4 E4B")
        print("   Modo: Edge Deployment (Offline)")

        # Initialize model (lazy loading for efficiency)
        self._model = None
        self._processor = None
        self._device = None

        print("Guardiao da Floresta pronto para proteger a Amazonia!")

    def _load_model(self):
        """Lazy load the Gemma model"""
        if self._model is not None:
            return

        try:
            from transformers import (
                Gemma4ForConditionalGeneration, 
                AutoProcessor,
                BitsAndBytesConfig
            )

            print("Carregando modelo Gemma 4 E4B...")

            # 4-bit quantization for edge deployment
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                self._model = Gemma4ForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                self._model = Gemma4ForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )

            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            self._device = next(self._model.parameters()).device
            print(f"Modelo carregado em: {self._device}")

        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            print("Usando modo simulacao para demonstracao...")
            self._model = None
            self._processor = None

    def _generate(self, prompt: str, images: Optional[List] = None, 
                  max_tokens: Optional[int] = None) -> str:
        """Generate text using Gemma"""
        self._load_model()

        if self._model is None:
            # Simulation mode for demo
            return self._simulate_response(prompt)

        max_tokens = max_tokens or self.config.max_new_tokens

        try:
            if images:
                # Multimodal generation
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img} for img in images
                    ] + [{"type": "text", "text": prompt}]
                }]

                inputs = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(self._device)
            else:
                # Text-only generation
                messages = [{"role": "user", "content": prompt}]
                inputs = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(self._device)

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=self.config.top_p
                )

            response = self._processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            print(f"Erro na geracao: {e}")
            return self._simulate_response(prompt)

    def _simulate_response(self, prompt: str) -> str:
        """Simulate responses when model is not available"""
        prompt_lower = prompt.lower()

        if "bioacoustic" in prompt_lower or "audio" in prompt_lower or "sound" in prompt_lower:
            return """BIOACOUSTIC ANALYSIS - SIMULATION

Detections:
- Native birds: 15 species identified
- Unusual pattern detected: possible human activity
- Frequencies analyzed: 200-8000 Hz
- Biodiversity index: 7.8/10 (Healthy)

Threats:
- Level: NORMAL (no chainsaw detection)
- Recommendation: Continue monitoring

Note: Simulation mode - Gemma 4 model not loaded"""

        elif "satellite" in prompt_lower or "image" in prompt_lower:
            return """SATELLITE IMAGE ANALYSIS - SIMULATION

Vegetation Cover:
- NDVI Average: 0.72 (Good)
- Area analyzed: ~500 hectares
- Changes detected: None significant

Alerts:
- Status: MONITORING
- No deforestation activity detected
- Existing roads: No expansion

Recommendations:
- Continue monthly monitoring
- Check dry seasons for changes

Note: Simulation mode - Gemma 4 model not loaded"""

        elif "agriculture" in prompt_lower or "farm" in prompt_lower or "cultivo" in prompt_lower:
            return """SUSTAINABLE AGRICULTURE ADVICE - SIMULATION

Recommendations for your region:
1. Implement agroforestry system (SAF)
2. Use drip irrigation to save water
3. Plant legumes as green manure
4. DO NOT burn - use composting
5. Maintain biodiversity corridors

Calendar (Acre):
- November-March: Rainy season - plant acai, cocoa
- June-September: Dry season - harvest, maintenance

Rights:
- Legal Reserve: 80% of property
- APPs: Protect springs and river margins

Note: Simulation mode - Gemma 4 model not loaded"""

        else:
            return """GUARDIAO DA FLORESTA - SIMULATION

Thank you for using Guardiao da Floresta! This is demo mode.

Available capabilities:
- Bioacoustic Monitoring
- Satellite Image Analysis  
- Sustainable Agriculture Assistant
- Biodiversity Educator

For full mode, load Gemma 4 E4B model.

Note: Simulation mode - Gemma 4 model not loaded"""

    # ========================================================================
    # AGENT 1: Bioacoustic Monitor
    # ========================================================================

    def analyze_bioacoustics(self, audio_features: Optional[Dict] = None,
                            audio_path: Optional[str] = None,
                            location: str = "Acre, Brasil",
                            duration: float = 60.0) -> Dict:
        """
        Bioacoustic Analysis Agent

        Analyzes forest audio to detect:
        - Chainsaws and heavy machinery (deforestation)
        - Invasive species
        - Biodiversity health indicators
        - Illegal mining activities
        """
        print(f"Analisando audio de {location}...")

        # Extract audio features if path provided
        if audio_path and audio_features is None:
            audio_features = self._extract_audio_features(audio_path)

        # Prepare analysis prompt
        features_desc = self._describe_audio_features(audio_features) if audio_features else "Dados nao disponiveis"

        system_prompt = """Voce e um especialista em bioacustica da floresta amazonica. 
Analise os dados de audio fornecidos e identifique:
1. Sinais de desmatamento (motosserras, maquinas pesadas)
2. Presenca de especies invasoras
3. Indicadores de saude da biodiversidade
4. Alertas urgentes necessarios

Responda em Portugues para comunidades locais e Ingles para ONGs."""

        prompt = f"""{system_prompt}

Dados da Gravacao:
- Localizacao: {location}
- Duracao: {duration} segundos
- Caracteristicas: {features_desc}

Base de Conhecimento:
- Especies conhecidas na regiao: {len(self.knowledge_base.species_db['birds'])} aves, 
  {len(self.knowledge_base.species_db['mammals'])} mamiferos
- Estacao atual: {self.knowledge_base.get_climate_for_region(location.split(',')[0].strip())}

Forneca uma analise detalhada com classificacao de ameaca."""

        response = self._generate(prompt)

        # Classify threat level
        threat_level, threat_score, keywords = self.threat_classifier.classify_text(response)

        result = {
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "duration_seconds": duration,
            "analysis": response,
            "threat_level": threat_level,
            "threat_score": threat_score,
            "matched_keywords": keywords,
            "audio_features": audio_features or {},
            "recommendations": self._generate_bioacoustic_recommendations(threat_level, keywords),
            "next_steps": self._generate_next_steps(threat_level)
        }

        print(f"   Analise completa - Nivel de ameaca: {threat_level}")
        return result

    def _extract_audio_features(self, audio_path: str) -> Dict:
        """Extract audio features using librosa"""
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=self.config.sample_rate)

            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft, hop_length=self.config.hop_length
            )

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)

            # RMS energy
            rms = librosa.feature.rms(y=y)

            # Chainsaw detection in frequency range
            chainsaw_band = self._detect_chainsaw_band(y, sr)

            return {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "mfcc_mean": float(np.mean(mfcc)),
                "mfcc_std": float(np.std(mfcc)),
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "zcr_mean": float(np.mean(zcr)),
                "rms_mean": float(np.mean(rms)),
                "chainsaw_band_energy": chainsaw_band,
                "dominant_frequencies": self._get_dominant_frequencies(y, sr)
            }

        except ImportError:
            print("   librosa nao instalado - usando features simuladas")
            return self._simulate_audio_features()
        except Exception as e:
            print(f"   Erro ao extrair features: {e}")
            return self._simulate_audio_features()

    def _simulate_audio_features(self) -> Dict:
        """Simulate audio features for demo"""
        return {
            "duration": 60.0,
            "sample_rate": 16000,
            "mfcc_mean": -15.2,
            "mfcc_std": 8.5,
            "spectral_centroid_mean": 1850.0,
            "spectral_rolloff_mean": 4200.0,
            "zcr_mean": 0.08,
            "rms_mean": 0.12,
            "chainsaw_band_energy": 0.15,
            "dominant_frequencies": [450, 1200, 2800],
            "note": "Simulated features for demonstration"
        }

    def _detect_chainsaw_band(self, y: np.ndarray, sr: int) -> float:
        """Detect energy in chainsaw frequency band (200-500Hz)"""
        # FFT
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), 1/sr)

        # Find indices for 200-500Hz
        low_idx = np.argmin(np.abs(freqs - 200))
        high_idx = np.argmin(np.abs(freqs - 500))

        # Calculate energy in band
        band_energy = np.sum(np.abs(fft[low_idx:high_idx])**2)
        total_energy = np.sum(np.abs(fft)**2)

        return float(band_energy / (total_energy + 1e-10))

    def _get_dominant_frequencies(self, y: np.ndarray, sr: int, n_peaks: int = 3) -> List[float]:
        """Get dominant frequencies in audio"""
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), 1/sr)

        # Only positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_fft = np.abs(fft[pos_mask])

        # Find peaks
        peak_indices = np.argsort(pos_fft)[-n_peaks:][::-1]
        return [float(pos_freqs[i]) for i in peak_indices]

    def _describe_audio_features(self, features: Dict) -> str:
        """Create text description of audio features"""
        return f"""Duracao: {features.get('duration', 'N/A')}s, 
Frequencia Dominante: {features.get('dominant_frequencies', ['N/A'])[0]:.0f}Hz,
Energia na banda de motosserra: {features.get('chainsaw_band_energy', 0):.3f},
Centroide Espectral: {features.get('spectral_centroid_mean', 0):.0f}Hz"""

    def _generate_bioacoustic_recommendations(self, threat_level: str, keywords: List[str]) -> List[str]:
        """Generate recommendations based on threat level"""
        recommendations = []

        if threat_level == "CRITICAL":
            recommendations.extend([
                "ATIVAR ALERTA IMEDIATO - Contatar IBAMA",
                "Enviar equipe de campo para verificacao",
                "Ativar cameras de vigilancia na area",
                "Notificar comunidades indigenas proximas"
            ])
        elif threat_level == "WARNING":
            recommendations.extend([
                "Aumentar frequencia de monitoramento",
                "Mapear atividades suspeitas na regiao",
                "Comparar com dados historicos da area"
            ])
        else:
            recommendations.extend([
                "Monitoramento regular - nenhuma ameaca detectada",
                "Continuar coleta de dados para baseline",
                "Registrar especies identificadas no banco de dados"
            ])

        return recommendations

    # ========================================================================
    # AGENT 2: Satellite Image Analyzer
    # ========================================================================

    def analyze_satellite_image(self, image: Union[str, Image.Image],
                               coordinates: Tuple[float, float],
                               date: Optional[str] = None) -> Dict:
        """
        Satellite Image Analysis Agent

        Analyzes satellite imagery to detect:
        - Deforestation patterns
        - Illegal mining operations
        - New road construction
        - Vegetation density changes
        """
        print(f"Analisando imagem de satelite em {coordinates}...")

        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB')

        # Calculate vegetation indices
        vegetation_data = self._calculate_vegetation_indices(img)

        # Prepare prompt for Gemma
        system_prompt = """Voce e um especialista em analise de imagens de satelite da Amazonia.
Analise a imagem fornecida e identifique:
1. Padroes de desmatamento
2. Operacoes de mineracao ilegal
3. Construcao de novas estradas
4. Mudancas na densidade vegetal
5. Areas de queimada

Forneca insights acionaveis para equipes de conservacao."""

        prompt = f"""{system_prompt}

Dados da Imagem:
- Coordenadas: {coordinates}
- Data: {date or datetime.now().strftime('%Y-%m-%d')}
- Indices de Vegetacao: {vegetation_data}
- Resolucao: {img.size}

Contexto Regional:
- {self.knowledge_base.get_climate_for_region('acre')}
- Areas de conservacao proximas: {list(self.knowledge_base.conservation_areas.keys())[:2]}

Analise detalhadamente e classifique o nivel de alerta."""

        response = self._generate(prompt, images=[img])

        # Classify alert status
        threat_level, threat_score, keywords = self.threat_classifier.classify_text(response)

        result = {
            "timestamp": datetime.now().isoformat(),
            "coordinates": coordinates,
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "analysis": response,
            "alert_status": threat_level,
            "threat_score": threat_score,
            "vegetation_indices": vegetation_data,
            "image_size": img.size,
            "recommendations": self._generate_satellite_recommendations(threat_level, vegetation_data),
            "action_plan": self._generate_action_plan(threat_level, coordinates)
        }

        print(f"   Analise completa - Status: {threat_level}")
        return result

    def _calculate_vegetation_indices(self, image: Image.Image) -> Dict:
        """Calculate vegetation indices from satellite image"""
        img_array = np.array(image).astype(float)

        # Extract bands (assuming RGB)
        red = img_array[:, :, 0]
        green = img_array[:, :, 1]
        nir = img_array[:, :, 2]  # Using blue as proxy for NIR in demo

        # NDVI (Normalized Difference Vegetation Index)
        ndvi = (nir - red) / (nir + red + 1e-10)

        # GNDVI (Green NDVI)
        gndvi = (nir - green) / (nir + green + 1e-10)

        # SAVI (Soil Adjusted Vegetation Index)
        L = 0.5
        savi = ((nir - red) * (1 + L)) / (nir + red + L + 1e-10)

        return {
            "ndvi_mean": float(np.mean(ndvi)),
            "ndvi_std": float(np.std(ndvi)),
            "ndvi_min": float(np.min(ndvi)),
            "ndvi_max": float(np.max(ndvi)),
            "gndvi_mean": float(np.mean(gndvi)),
            "savi_mean": float(np.mean(savi)),
            "vegetation_health": self._classify_ndvi(float(np.mean(ndvi)))
        }

    def _classify_ndvi(self, ndvi_mean: float) -> str:
        """Classify vegetation health from NDVI"""
        if ndvi_mean > 0.6:
            return "EXCELENTE - Densa cobertura florestal"
        elif ndvi_mean > 0.4:
            return "BOA - Cobertura vegetal saudavel"
        elif ndvi_mean > 0.2:
            return "MODERADA - Possivel degradacao"
        elif ndvi_mean > 0.0:
            return "RUIM - Significativa perda vegetal"
        else:
            return "CRITICA - Area desmatada ou agua"

    def _generate_satellite_recommendations(self, threat_level: str, 
                                           vegetation_data: Dict) -> List[str]:
        """Generate satellite-specific recommendations"""
        recommendations = []

        ndvi = vegetation_data.get("ndvi_mean", 0)

        if threat_level == "CRITICAL":
            recommendations.extend([
                "ALERTA VERMELHO - Desmatamento ativo detectado",
                "Acionar fiscalizacao aerea (IBAMA/ICMBio)",
                "Solicitar imagens de alta resolucao (Planet/Maxar)",
                "Preparar processo de embargo da area"
            ])
        elif ndvi < 0.3:
            recommendations.extend([
                "NDVI baixo detectado - Investigar causa",
                "Verificar se e desmatamento recente ou estacao seca",
                "Comparar com imagens historicas da area"
            ])
        else:
            recommendations.extend([
                "Cobertura vegetal saudavel mantida",
                "Monitorar tendencias de longo prazo",
                "Area prioritaria para conservacao"
            ])

        return recommendations

    def _generate_action_plan(self, threat_level: str, coordinates: Tuple[float, float]) -> str:
        """Generate specific action plan"""
        lat, lon = coordinates

        if threat_level == "CRITICAL":
            return f"""PLANO DE ACAO IMEDIATA
1. Coordenadas: {lat:.4f}, {lon:.4f}
2. Contato IBAMA: 0800-61-8080
3. Coordenar com policia federal ambiental
4. Documentar evidencias fotograficas
5. Notificar comunidades locais em raio de 5km
6. Solicitar sobrevoo de verificacao"""

        elif threat_level == "WARNING":
            return f"""PLANO DE MONITORAMENTO INTENSIVO
1. Coordenadas: {lat:.4f}, {lon:.4f}
2. Aumentar frequencia de imagens para semanal
3. Instalar cameras de trilha na regiao
4. Capacitar vigilantes comunitarios
5. Preparar relatorio para orgaos ambientais"""

        return f"""PLANO DE CONSERVACAO CONTINUA
1. Coordenadas: {lat:.4f}, {lon:.4f}
2. Manter monitoramento mensal
3. Documentar biodiversidade da area
4. Engajar comunidades locais na protecao
5. Considerar inclusao em area protegida"""

    # ========================================================================
    # AGENT 3: Sustainable Agriculture Advisor
    # ========================================================================

    def agriculture_advisor(self, query: str, 
                           farmer_context: Optional[Dict] = None) -> Dict:
        """
        Sustainable Agriculture Advisor Agent

        Provides offline-first farming advice for Amazon small farmers:
        - Sustainable farming practices
        - Crop calendar and recommendations
        - Deforestation prevention
        - Soil conservation
        - Legal compliance
        """
        print(f"Consulta agricola: {query[:50]}...")

        context = farmer_context or {}
        location = context.get("location", "Acre, Brasil")
        crop = context.get("crop", "Misto")
        farm_size = context.get("size", "Pequeno")
        language = context.get("lang", "pt-BR")
        experience = context.get("experience", "Intermediario")

        # Get local knowledge
        climate = self.knowledge_base.get_climate_for_region(location.split(',')[0].strip())
        crop_info = self.knowledge_base.agriculture_guide["crops"].get(crop.lower(), {})

        system_prompt = f"""Voce e um especialista em agricultura sustentavel para a Amazonia.
Forneca conselhos praticos, ecologicos e culturalmente apropriados para pequenos produtores.

Principios:
- Prevenir desmatamento
- Respeitar terras indigenas
- Usar recursos locais eficientemente
- Promover renda sustentavel
- Preservar biodiversidade

Responda em Portugues brasileiro simples e acessivel."""

        prompt = f"""{system_prompt}

Perfil do Produtor:
- Localizacao: {location}
- Tipo de cultivo: {crop}
- Tamanho da propriedade: {farm_size}
- Experiencia: {experience}
- Idioma preferido: {language}

Contexto Local:
- Clima: {climate}
- Informacoes da cultura: {crop_info}
- Praticas sustentaveis disponiveis: {self.knowledge_base.agriculture_guide["sustainable_practices"]}
- Conservacao do solo: {self.knowledge_base.agriculture_guide["soil_conservation"]}

Pergunta do Produtor:
{query}

Forneca uma resposta completa com:
1. Resposta direta a pergunta
2. Praticas recomendadas passo a passo
3. Calendario agricola relevante
4. Avisos legais importantes
5. Recursos adicionais"""

        response = self._generate(prompt, max_tokens=800)

        result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "farmer_context": context,
            "response": response,
            "local_climate": climate,
            "crop_info": crop_info,
            "legal_reminders": self._generate_legal_reminders(location),
            "sustainable_practices": self.knowledge_base.agriculture_guide["sustainable_practices"][:3],
            "follow_up_questions": self._generate_follow_up_questions(query, crop)
        }

        print(f"   Conselho agricola gerado")
        return result

    def _generate_legal_reminders(self, location: str) -> List[str]:
        """Generate legal compliance reminders"""
        return [
            "Reserva Legal: Manter 80% da propriedade preservada (Amazonia)",
            "APP: Proteger nascentes, margens de rios e topo de morros",
            "CAR: Cadastro Ambiental Rural obrigatorio",
            "NAO desmatar fora da area permitida (20% no maximo)",
            "Respeitar Terras Indigenas e Unidades de Conservacao vizinhas"
        ]

    def _generate_follow_up_questions(self, query: str, crop: str) -> List[str]:
        """Generate follow-up questions for the farmer"""
        return [
            f"Voce gostaria de saber mais sobre rotacao de culturas com {crop}?",
            "Como esta a qualidade do solo da sua propriedade?",
            "Voce ja considerou implementar um Sistema Agroflorestal (SAF)?",
            "Precisa de informacoes sobre acesso a credito rural sustentavel?"
        ]

    # ========================================================================
    # AGENT 4: Biodiversity Educator
    # ========================================================================

    def biodiversity_educator(self, query: str,
                             species_image: Optional[Image.Image] = None,
                             user_age: int = 12,
                             language: str = "pt-BR") -> Dict:
        """
        Biodiversity Education Agent

        Interactive learning tool for Amazon biodiversity:
        - Species identification from images
        - Fun facts and conservation status
        - Quiz generation
        - Citizen science actions
        - Age-appropriate content
        """
        print(f"Educador de biodiversidade - Idade: {user_age}...")

        # Adjust content complexity based on age
        complexity = "simple" if user_age < 10 else "detailed" if user_age < 15 else "advanced"

        system_prompt = f"""Voce e um educador de biodiversidade amazonica engajador e apaixonado.
Crie conteudo educativo simples e ludico
para jovens de {user_age} anos sobre a fauna e flora da Amazonia.

Objetivos:
1. Despertar amor pela natureza
2. Ensinar sobre conservacao
3. Incentivar ciencia cidada
4. Respeitar conhecimento tradicional indigena

Responda em Portugues brasileiro muito simples."""

        prompt = f"""{system_prompt}

Pergunta do Estudante:
{query}

Contexto:
- Idade: {user_age} anos
- Nivel: {complexity}
- Especies conhecidas na regiao: {sum(len(v) for v in self.knowledge_base.species_db.values())}

Instrucoes:
1. Use analogias simples e linguagem ludica
2. Mencione o status de conservacao das especies
3. Explique por que a conservacao e importante
4. Sugira acoes praticas que o jovem pode tomar
5. Inclua um pequeno jogo ou desafio

Crie uma experiencia de aprendizado memoravel!"""

        images = [species_image] if species_image else None
        response = self._generate(prompt, images=images, max_tokens=700)

        # Generate quiz based on response
        quiz = self._generate_quiz(response, user_age)

        # Generate conservation actions
        actions = self._generate_conservation_actions(user_age)

        result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "user_age": user_age,
            "complexity_level": complexity,
            "educational_content": response,
            "quiz": quiz,
            "conservation_actions": actions,
            "species_identification": self._identify_species_from_text(response),
            "next_learning_topics": self._suggest_next_topics(query)
        }

        print(f"   Conteudo educacional gerado")
        return result

    def _generate_quiz(self, content: str, age: int) -> List[Dict]:
        """Generate age-appropriate quiz questions"""
        if age < 10:
            return [
                {
                    "question": "Qual e o maior animal da Amazonia?",
                    "options": ["Onca-pintada", "Boto-cor-de-rosa", "Ariranha", "Preguica"],
                    "correct": 1,
                    "fun_fact": "O boto-cor-de-rosa pode chegar a 2.5 metros de comprimento!"
                },
                {
                    "question": "Qual planta e super importante para os indigenas?",
                    "options": ["Rosa", "Acai", "Girassol", "Cacto"],
                    "correct": 1,
                    "fun_fact": "O acai e chamado de 'ouro roxo' da Amazonia!"
                }
            ]
        else:
            return [
                {
                    "question": "Qual indice mede a saude da vegetacao em imagens de satelite?",
                    "options": ["NDVI", "GPS", "RGB", "API"],
                    "correct": 0,
                    "explanation": "NDVI (Normalized Difference Vegetation Index) varia de -1 a 1, onde valores proximos a 1 indicam vegetacao saudavel."
                },
                {
                    "question": "Qual e a porcentagem minima de Reserva Legal na Amazonia?",
                    "options": ["50%", "65%", "80%", "100%"],
                    "correct": 2,
                    "explanation": "Na Amazonia Legal, 80% da propriedade deve ser mantida como Reserva Legal."
                }
            ]

    def _generate_conservation_actions(self, age: int) -> List[str]:
        """Generate age-appropriate conservation actions"""
        if age < 10:
            return [
                "Plantar uma arvore nativa no seu quintal",
                "Tirar fotos de passaros e aprender seus nomes",
                "Economizar agua sempre que possivel",
                "Contar para seus amigos sobre a Amazonia"
            ]
        elif age < 15:
            return [
                "Participar de projetos de ciencia cidada (eBird, iNaturalist)",
                "Criar um jardim de plantas nativas",
                "Monitorar a qualidade da agua do rio local",
                "Fazer videos educativos sobre conservacao"
            ]
        else:
            return [
                "Participar de pesquisas cientificas locais",
                "Organizar campanhas de conscientizacao na escola",
                "Voluntariar em projetos de reflorestamento",
                "Desenvolver apps ou ferramentas para monitoramento ambiental"
            ]

    def _identify_species_from_text(self, text: str) -> List[Dict]:
        """Extract species mentions from text"""
        identified = []
        for category, species in self.knowledge_base.species_db.items():
            for name, info in species.items():
                if name.replace("_", " ") in text.lower():
                    identified.append({
                        "name": name.replace("_", " ").title(),
                        "category": category,
                        **info
                    })
        return identified

    def _suggest_next_topics(self, query: str) -> List[str]:
        """Suggest next learning topics"""
        topics = [
            "Polinizadores da Amazonia",
            "Sistemas Agroflorestais",
            "Ciclo da Agua na Floresta Tropical",
            "Conhecimento Tradicional Indigena",
            "Biotecnologia da Biodiversidade",
            "Mudancas Climaticas e a Amazonia"
        ]
        return topics[:4]

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _generate_next_steps(self, threat_level: str) -> List[str]:
        """Generate next steps based on threat level"""
        if threat_level == "CRITICAL":
            return [
                "Alertar autoridades em menos de 1 hora",
                "Documentar com fotos e coordenadas GPS",
                "Coordenar com comunidades locais",
                "Preparar relatorio para midia"
            ]
        elif threat_level == "WARNING":
            return [
                "Agendar visita de campo em 48h",
                "Revisar dados historicos da area",
                "Consultar imagens de satelite mais recentes",
                "Informar gestores de unidades de conservacao"
            ]
        return [
            "Manter rotina de monitoramento",
            "Atualizar baseline de biodiversidade",
            "Compartilhar dados com rede de pesquisa"
        ]

    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            "agent_name": "Guardiao da Floresta",
            "version": "1.0.0",
            "model": self.config.model_name,
            "device": str(self._device) if self._device else "Not loaded",
            "knowledge_base_species": sum(len(v) for v in self.knowledge_base.species_db.values()),
            "conservation_areas": len(self.knowledge_base.conservation_areas),
            "status": "Online" if self._model is not None else "Simulation Mode",
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# Demo & Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GUARDIAO DA FLORESTA - Demonstracao")
    print("=" * 70)

    # Initialize agent
    guardiao = GuardiaoDaFloresta()

    # Test 1: Bioacoustic Analysis
    print("\n" + "=" * 70)
    print("TESTE 1: Analise Bioacustica")
    print("=" * 70)
    bio_result = guardiao.analyze_bioacoustics(
        location="Reserva Chico Mendes, Acre",
        duration=120.0
    )
    print(f"\nNivel de Ameaca: {bio_result['threat_level']}")
    print(f"Recomendacoes: {bio_result['recommendations'][0]}")

    # Test 2: Satellite Analysis
    print("\n" + "=" * 70)
    print("TESTE 2: Analise de Satelite")
    print("=" * 70)

    # Create a demo satellite image
    demo_img = Image.new('RGB', (512, 512), color=(34, 139, 34))
    sat_result = guardiao.analyze_satellite_image(
        image=demo_img,
        coordinates=(-9.0238, -70.8120),
        date="2026-07-22"
    )
    print(f"\nStatus de Alerta: {sat_result['alert_status']}")
    print(f"NDVI: {sat_result['vegetation_indices']['ndvi_mean']:.3f}")

    # Test 3: Agriculture Advisor
    print("\n" + "=" * 70)
    print("TESTE 3: Assistente Agricola")
    print("=" * 70)
    agri_result = guardiao.agriculture_advisor(
        query="Como posso aumentar a producao de acai sem desmatar?",
        farmer_context={
            "location": "Rio Branco, Acre",
            "crop": "Acai",
            "size": "Medio (50 hectares)",
            "experience": "Avancado"
        }
    )
    print(f"\nResposta: {agri_result['response'][:200]}...")

    # Test 4: Biodiversity Education
    print("\n" + "=" * 70)
    print("TESTE 4: Educador de Biodiversidade")
    print("=" * 70)
    edu_result = guardiao.biodiversity_educator(
        query="Me conta sobre a onca-pintada!",
        user_age=10
    )
    print(f"\nConteudo: {edu_result['educational_content'][:200]}...")
    print(f"Quiz: {edu_result['quiz'][0]['question']}")

    # System Status
    print("\n" + "=" * 70)
    print("STATUS DO SISTEMA")
    print("=" * 70)
    status = guardiao.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Demonstracao completa!")
    print("Guardiao da Floresta pronto para proteger a Amazonia!")
    print("=" * 70)
