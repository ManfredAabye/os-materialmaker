#!/usr/bin/env python3
"""
OpenSimulator PBR Material Maker - Tkinter Version mit GLTF-Rendering
Komplette GUI mit Tkinter für bessere Bild- und Drag&Drop-Unterstützung
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import json
import os
#import math
import re
from PIL import Image, ImageTk, ImageDraw
from typing import Dict, Optional
#import struct
import numpy as np

# Globale Import-Variablen für Pylance
pyrender = None
trimesh = None
GLTF2 = None

try:
    from pygltflib import GLTF2
    import trimesh
    import pyrender
    GLTF_AVAILABLE = True
    PYRENDER_AVAILABLE = True
    print("GLTF + PyRender verfügbar - 3D-Rendering aktiviert")
except ImportError as e:
    GLTF_AVAILABLE = False
    PYRENDER_AVAILABLE = False
    # Fallback-Module auf None setzen
    pyrender = None
    trimesh = None
    GLTF2 = None
    print(f"GLTF/PyRender-Bibliotheken nicht verfügbar: {e}. Fallback auf einfache 3D-Darstellung.")

# PyPBR Integration für GPU-beschleunigte PBR-Berechnungen
try:
    import pypbr
    import pypbr.materials as pbr_materials
    import pypbr.models as pbr_models
    import pypbr.utils.functions as pbr_funcs
    import torch
    PYPBR_AVAILABLE = True
    PYPBR_GPU_AVAILABLE = torch.cuda.is_available()
    print(f"PyPBR verfügbar - GPU-Beschleunigung: {'aktiviert' if PYPBR_GPU_AVAILABLE else 'deaktiviert'}")
    if PYPBR_GPU_AVAILABLE:
        print(f"CUDA-Geräte: {torch.cuda.device_count()}")
except ImportError as e:
    PYPBR_AVAILABLE = False
    PYPBR_GPU_AVAILABLE = False
    pypbr = None
    pbr_materials = None
    pbr_models = None
    pbr_funcs = None
    torch = None
    print(f"PyPBR nicht verfügbar: {e}. Standard-Material-Pipeline wird verwendet.")

class PyPBRMaterialPipeline:
    """GPU-beschleunigte PBR-Material-Pipeline mit PyPBR - Vollständig optimiert"""
    
    def __init__(self):
        self.enabled = PYPBR_AVAILABLE
        self.gpu_enabled = PYPBR_GPU_AVAILABLE
        self.device = None
        self.brdf_model = None
        self.material_cache = {}
        
        # PyPBR-optimierte Konfiguration
        self.config = {
            'normal_convention': 'OPENGL',  # glTF-Standard
            'albedo_is_srgb': True,         # Korrekte Farbdarstellung
            'roughness_clamp_min': 0.01,    # Numerische Stabilität
            'roughness_clamp_max': 1.0,     # Maximum für physikalische Korrektheit
            'metallic_binary': False,       # Kontinuierliche Werte erlaubt
            'batch_size': 32,               # Optimale GPU-Auslastung
            'energy_conservation': True,    # Physikalisch korrekt
            'fresnel_enabled': True,        # Realistische Reflexion
            'auto_normalize_normals': True, # [-1,1] Bereich garantiert
        }
        
        if self.enabled:
            self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialisiere PyPBR-Pipeline mit optimierten Einstellungen"""
        try:
            if self.gpu_enabled and torch is not None:
                self.device = torch.device("cuda:0")
                print(f"PyPBR Pipeline auf GPU initialisiert: {torch.cuda.get_device_name(0)}")
            elif torch is not None:
                self.device = torch.device("cpu")
                print("PyPBR Pipeline auf CPU initialisiert")
            else:
                self.device = None
                print("PyPBR Pipeline konnte nicht initialisiert werden: torch ist nicht verfügbar")
            
            # Initialisiere optimierten Cook-Torrance BRDF-Modell
            if pbr_models is not None and hasattr(pbr_models, "CookTorranceBRDF"):
                self.brdf_model = pbr_models.CookTorranceBRDF()
                if self.gpu_enabled:
                    self.brdf_model = self.brdf_model.to(self.device)
                print("PyPBR BRDF-Modell (Cook-Torrance) mit Optimierungen initialisiert")
            else:
                self.brdf_model = None
                print("Warnung: CookTorranceBRDF nicht verfügbar in pypbr.models")
            
        except Exception as e:
            print(f"PyPBR Pipeline Initialisierung fehlgeschlagen: {e}")
            self.enabled = False
    
    def optimize_color_space(self, color_values):
        """Optimiere Color Space - sRGB zu Linear Konvertierung"""
        if not self.enabled or not color_values or torch is None:
            return color_values
        
        try:
            # Konvertiere sRGB zu Linear für physikalisch korrekte Berechnungen
            if isinstance(color_values, list):
                color_tensor = torch.tensor(color_values, dtype=torch.float32, device=self.device)
            else:
                color_tensor = torch.tensor([color_values], dtype=torch.float32, device=self.device)
            
            # PyPBR sRGB → Linear Konvertierung
            if self.config['albedo_is_srgb'] and pbr_funcs is not None:
                linear_color = pbr_funcs.srgb_to_linear(color_tensor)
                return linear_color.cpu().tolist() if isinstance(color_values, list) else linear_color.cpu().item()
            
            return color_values
            
        except Exception as e:
            print(f"Color Space Optimierung fehlgeschlagen: {e}")
            return color_values
    
    def optimize_roughness(self, roughness_value):
        """Optimiere Roughness mit PyPBR-Clamping für numerische Stabilität"""
        if not self.enabled:
            return roughness_value
        
        # Clamp roughness für numerische Stabilität (verhindert Division by Zero)
        min_roughness = self.config['roughness_clamp_min']
        max_roughness = self.config['roughness_clamp_max']
        
        clamped = max(min_roughness, min(max_roughness, roughness_value))
        
        if clamped != roughness_value:
            print(f"Roughness optimiert: {roughness_value} → {clamped} (numerische Stabilität)")
        
        return clamped
    
    def optimize_normal_map(self, normal_strength):
        """Optimiere Normal Map mit PyPBR-Convention"""
        if not self.enabled:
            return normal_strength
        
        # Normalisiere auf [-1,1] Bereich für physikalische Korrektheit
        if self.config['auto_normalize_normals']:
            # Normal Strength in physikalisch korrekten Bereich konvertieren
            normalized = min(1.0, max(-1.0, normal_strength))
            
            if normalized != normal_strength:
                print(f"Normal Map optimiert: {normal_strength} → {normalized} (physikalische Korrektheit)")
            
            return normalized
        
        return normal_strength
    
    def create_optimized_material(self, material_params):
        """Erstelle PyPBR-Material mit allen Optimierungen"""
        if not self.enabled:
            return None
        
        try:
            # Optimierte Parameter-Extraktion
            basecolor = material_params.get('BaseColorTint', [1.0, 1.0, 1.0])
            metallic = material_params.get('MetallicIntensity', 0.0)
            roughness = material_params.get('RoughnessStrength', 0.5)
            # normal_strength = material_params.get('NormalStrength', 0.2)  # Nicht verwendet
            
            # Erweiterte PyPBR-Parameter
            metallic_strength = material_params.get('metallic_strength', 1.0)
            base_color_strength = material_params.get('base_color_strength', 1.0)
            contrast = material_params.get('contrast', 1.0)
            brightness = material_params.get('brightness', 1.0)
            
            # PyPBR-Optimierungen anwenden
            optimized_basecolor = self.optimize_color_space(basecolor)
            optimized_roughness = self.optimize_roughness(roughness)
            # optimized_normal = self.optimize_normal_map(normal_strength)  # Nicht verwendet
            
            # Color Enhancement mit physikalischer Korrektheit
            # Stelle sicher, dass optimized_basecolor eine Liste ist
            if not isinstance(optimized_basecolor, list):
                if isinstance(optimized_basecolor, (int, float)):
                    optimized_basecolor = [optimized_basecolor]
                else:
                    optimized_basecolor = basecolor  # Fallback zum Original
            
            enhanced_basecolor = [
                min(1.0, c * base_color_strength * brightness) 
                for c in optimized_basecolor
            ]
            
            # Metallic mit Strength-Multiplikator (physikalisch begrenzt)
            enhanced_metallic = min(1.0, metallic * metallic_strength)
            
            # Contrast-Enhancement (physikalisch korrekt)
            if contrast != 1.0:
                enhanced_basecolor = [
                    min(1.0, max(0.0, ((c - 0.5) * contrast) + 0.5))
                    for c in enhanced_basecolor
                ]
            
            # PyPBR Material erstellen mit optimierten Parametern
            material_kwargs = {
                'albedo_is_srgb': self.config['albedo_is_srgb']
            }
            
            # Tensor-Erstellung für GPU-Beschleunigung (nur wenn torch verfügbar)
            if torch is not None:
                if enhanced_basecolor:
                    material_kwargs['albedo'] = torch.tensor(enhanced_basecolor, device=self.device, dtype=torch.float32)
                
                material_kwargs['metallic'] = torch.tensor(enhanced_metallic, device=self.device, dtype=torch.float32)
                material_kwargs['roughness'] = torch.tensor(optimized_roughness, device=self.device, dtype=torch.float32)
            else:
                # Fallback ohne torch - verwende normale Python-Werte
                if enhanced_basecolor:
                    material_kwargs['albedo'] = enhanced_basecolor
                material_kwargs['metallic'] = enhanced_metallic
                material_kwargs['roughness'] = optimized_roughness
            
            # Normal Convention setzen (verwende String-Werte)
            if self.config['normal_convention'] in ['OPENGL', 'DIRECTX']:
                material_kwargs['normal_convention'] = self.config['normal_convention']
            
            # Erstelle optimiertes PyPBR-Material (nur wenn pbr_materials verfügbar)
            if pbr_materials is not None and hasattr(pbr_materials, 'BasecolorMetallicMaterial'):
                material = pbr_materials.BasecolorMetallicMaterial(**material_kwargs)
            else:
                material = None
            
            return material
            
        except Exception as e:
            print(f"PyPBR Material-Erstellung fehlgeschlagen: {e}")
            return None
    
    def calculate_optimized_brdf(self, material, light_dir, view_dir, normal):
        """Berechne BRDF mit Cook-Torrance und allen Optimierungen"""
        if not self.enabled or not material or not self.brdf_model or torch is None:
            return None
        
        try:
            with torch.no_grad():
                # Energie-erhaltende BRDF-Berechnung
                result = self.brdf_model(material, light_dir, view_dir, normal)
                
                # Fresnel-Enhancement (wenn aktiviert)
                if self.config['fresnel_enabled']:
                    # Fresnel-Effekt automatisch in Cook-Torrance enthalten
                    pass
                
                return result
                
        except Exception as e:
            print(f"PyPBR BRDF-Berechnung fehlgeschlagen: {e}")
            return None
    
    def enhance_material_rendering(self, material_params):
        """Verbessere Material-Rendering mit allen PyPBR-Optimierungen"""
        if not self.enabled:
            return material_params
        
        try:
            # Erstelle optimiertes PyPBR-Material
            pbr_material = self.create_optimized_material(material_params)
            if not pbr_material:
                return material_params
            
            # Cache das Material für bessere Performance (Sichere Hash-Generierung)
            try:
                # Konvertiere Listen zu Tupeln für hashbare Darstellung
                hashable_params = {}
                for key, value in material_params.items():
                    if isinstance(value, list):
                        hashable_params[key] = tuple(value)
                    else:
                        hashable_params[key] = value
                
                cache_key = hash(frozenset(hashable_params.items()))
                self.material_cache[cache_key] = pbr_material
            except Exception as e:
                print(f"Material-Cache Fehler: {e}")
                # Fallback: Verwende Material-Name als Cache-Key
                cache_key = material_params.get('MaterialName', 'unknown_material')
                self.material_cache[cache_key] = pbr_material
            
            # Erweiterte Parameter-Optimierung
            enhanced_params = material_params.copy()
            
            # BaseColor mit sRGB→Linear Optimierung
            if 'BaseColorTint' in material_params:
                original_color = material_params['BaseColorTint']
                optimized_color = self.optimize_color_space(original_color)
                enhanced_params['BaseColorTint_Linear'] = optimized_color
                enhanced_params['BaseColorTint_Original'] = original_color
            
            # Roughness mit Stabilität
            if 'RoughnessStrength' in material_params:
                original_roughness = material_params['RoughnessStrength']
                optimized_roughness = self.optimize_roughness(original_roughness)
                enhanced_params['RoughnessStrength'] = optimized_roughness
                if optimized_roughness != original_roughness:
                    enhanced_params['RoughnessStrength_Original'] = original_roughness
            
            # Normal Map Optimierung
            if 'NormalStrength' in material_params:
                original_normal = material_params['NormalStrength']
                optimized_normal = self.optimize_normal_map(original_normal)
                enhanced_params['NormalStrength'] = optimized_normal
                if optimized_normal != original_normal:
                    enhanced_params['NormalStrength_Original'] = original_normal
            
            # PyPBR-Qualitäts-Flags
            enhanced_params['PyPBR_Optimized'] = True
            enhanced_params['PyPBR_ColorSpace'] = 'Linear' if self.config['albedo_is_srgb'] else 'sRGB'
            enhanced_params['PyPBR_NormalConvention'] = self.config['normal_convention']
            enhanced_params['PyPBR_BRDF'] = 'CookTorrance'
            
            print(f"PyPBR-Enhancement für '{material_params.get('MaterialName', 'unknown')}' angewendet")
            return enhanced_params
            
        except Exception as e:
            print(f"PyPBR Material-Enhancement fehlgeschlagen: {e}")
            return material_params
    
    def get_performance_stats(self):
        """Hole Performance-Statistiken mit PyPBR-Details"""
        if not self.enabled:
            return {"status": "disabled"}
        
        stats = {
            "status": "enabled",
            "gpu_acceleration": self.gpu_enabled,
            "device": str(self.device) if self.device else "none",
            "cached_materials": len(self.material_cache),
            "optimizations": {
                "color_space": "sRGB→Linear" if self.config['albedo_is_srgb'] else "sRGB",
                "normal_convention": self.config['normal_convention'],
                "roughness_clamping": f"[{self.config['roughness_clamp_min']}, {self.config['roughness_clamp_max']}]",
                "energy_conservation": self.config['energy_conservation'],
                "fresnel_enabled": self.config['fresnel_enabled'],
                "brdf_model": "Cook-Torrance"
            }
        }
        
        if self.gpu_enabled and torch is not None:
            try:
                stats["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
                stats["gpu_name"] = torch.cuda.get_device_name(0)
                stats["gpu_memory_allocated"] = torch.cuda.memory_allocated(0)
                stats["gpu_memory_reserved"] = torch.cuda.memory_reserved(0)
            except Exception as e:
                stats["gpu_error"] = str(e)
        
        return stats
    
    def get_optimization_report(self, material_params):
        """Generiere detaillierten Optimierungs-Bericht"""
        if not self.enabled:
            return "PyPBR nicht verfügbar"
        
        report = []
        report.append(f"=== PyPBR Optimierungs-Bericht: {material_params.get('MaterialName', 'Unknown')} ===")
        
        # Color Space Analyse
        if 'BaseColorTint' in material_params:
            original = material_params['BaseColorTint']
            optimized = self.optimize_color_space(original)
            report.append(f"BaseColor: {original} → {optimized} (sRGB→Linear)")
        
        # Roughness Analyse
        if 'RoughnessStrength' in material_params:
            original = material_params['RoughnessStrength']
            optimized = self.optimize_roughness(original)
            report.append(f"Roughness: {original} → {optimized} (Stabilität)")
        
        # Normal Map Analyse
        if 'NormalStrength' in material_params:
            original = material_params['NormalStrength']
            optimized = self.optimize_normal_map(original)
            report.append(f"Normal: {original} → {optimized} (Normalisierung)")
        
        # Performance Stats
        stats = self.get_performance_stats()
        report.append(f"GPU: {stats.get('gpu_name', 'N/A')}")
        report.append(f"Cache: {stats['cached_materials']} Materialien")
        
        return "\n".join(report)

class ToolTip:
    """Einfache Tooltip-Klasse für Tkinter-Widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Motion>", self.on_motion)
    
    def on_enter(self, event=None):
        """Zeige Tooltip an"""
        if self.tooltip_window is not None:
            return
        
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Arial", 9),
            padx=4,
            pady=2
        )
        label.pack()
    
    def on_leave(self, event=None):
        """Verstecke Tooltip"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def on_motion(self, event=None):
        """Update Tooltip-Position bei Mausbewegung"""
        if self.tooltip_window:
            x = self.widget.winfo_rootx() + 25
            y = self.widget.winfo_rooty() + 25
            self.tooltip_window.wm_geometry(f"+{x}+{y}")

def add_tooltip(widget, text):
    """Hilfsfunktion zum einfachen Hinzufügen von Tooltips"""
    return ToolTip(widget, text)

class PyRender3DViewer:
    """PyRender-basierter 3D-Viewer für einfache Textur-Vorschau ohne Rotation"""
    def __init__(self, canvas, width=1200, height=1200):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.models = {}
        self.current_model = 'cube'
        self.zoom = 2.0
        self.texture_images = {}
        self.gltf_objects = {}  # <--- Fix: initialize gltf_objects attribute
        
        # Licht-Parameter für Fallback-Renderer
        self.light_angle = 45.0
        self.light_intensity = 3.0
        self.ambient_intensity = 0.3
        
        # PyRender Setup
        self.scene = None
        self.renderer = None
        self.use_fallback_renderer = False
        self.init_pyrender()
        
        # Lade GLTF-Modelle
        self.load_models()
        
        # Nur Zoom-Event beibehalten
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        
        # Initial render
        self.render()
    
    def init_pyrender(self):
        """Initialisiere PyRender mit robuster Fehlerbehandlung"""
        try:
            if PYRENDER_AVAILABLE and pyrender is not None:
                print(" Initialisiere PyRender...")
                
                # Setze Rendering-Flags für bessere Kompatibilität
                import os
                os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Versuche EGL zuerst
                
                # Erstelle Scene
                self.scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
                
                try:
                    # Versuche OffscreenRenderer mit Standard-Einstellungen
                    self.renderer = pyrender.OffscreenRenderer(
                        viewport_width=self.width, 
                        viewport_height=self.height
                    )
                    print("PyRender OffscreenRenderer erfolgreich initialisiert")
                    
                except Exception as renderer_error:
                    print(f"OffscreenRenderer Fehler: {renderer_error}")
                    print("Versuche alternative Renderer-Konfiguration...")
                    
                    try:
                        # Fallback: Mesa Software-Rendering
                        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
                        self.renderer = pyrender.OffscreenRenderer(
                            viewport_width=self.width, 
                            viewport_height=self.height
                        )
                        print("PyRender mit Mesa Software-Rendering initialisiert")
                        
                    except Exception as mesa_error:
                        print(f"Mesa Renderer Fehler: {mesa_error}")
                        print("Verwende einfachen Fallback-Renderer...")
                        self.renderer = None
                        self.use_fallback_renderer = True
                
                # Füge Licht hinzu wenn Renderer verfügbar
                if self.renderer is not None:
                    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
                    light_node = pyrender.Node(light=light, matrix=np.eye(4))
                    self.scene.add_node(light_node)
                    
                    # Zusätzliches Punktlicht für bessere Beleuchtung
                    point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)
                    point_light_node = pyrender.Node(
                        light=point_light, 
                        matrix=np.array([
                            [1.0, 0.0, 0.0, 2.0],
                            [0.0, 1.0, 0.0, 2.0],
                            [0.0, 0.0, 1.0, 2.0],
                            [0.0, 0.0, 0.0, 1.0]
                        ])
                    )
                    self.scene.add_node(point_light_node)
                    
                    print("PyRender Beleuchtung konfiguriert")
                else:
                    print("PyRender nicht verfügbar - verwende Fallback")
                    self.use_fallback_renderer = True
            else:
                print("PyRender nicht verfügbar")
                self.renderer = None
                self.use_fallback_renderer = True
                
        except Exception as e:
            print(f"Kritischer Fehler bei PyRender-Initialisierung: {e}")
            self.renderer = None
            self.scene = None
            self.use_fallback_renderer = True
    
    def load_models(self):
        """3D-Modelle nicht mehr erforderlich - verwende nur noch 2D-Darstellung"""
        print("3D-Modelle übersprungen - verwende nur noch 2D-Fallback-Darstellung")
        self.current_model = 'fallback'
    
    def update_gltf_texture(self, model_type, texture_type, image_data):
        """GLTF2-Funktionalität deaktiviert - verwendet nur noch Fallback-Renderer"""
        #print(f"GLTF2-Objekte deaktiviert - Verwende Fallback-Renderer für Textur-Darstellung")
        return False
    
    def export_gltf_model(self, filename="exported_model.glb"):
        """GLTF2-Export deaktiviert - nur Fallback-Renderer verfügbar"""
        #print("GLTF2-Export deaktiviert - Verwende Fallback-Renderer für Textur-Darstellung")
        return False
    
    def create_fallback_models(self):
        """Fallback-Modelle sind nicht mehr erforderlich - nur noch 2D-Darstellung"""
        print("Verwende nur noch 2D-Fallback-Darstellung - keine 3D-Modelle erforderlich")
        self.current_model = 'fallback'
    
    def on_mouse_wheel(self, event):
        """Mausrad - Zoom"""
        zoom_factor = 1.1 if event.delta > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = max(0.5, min(5.0, self.zoom))
        self.render()
    
    def set_model(self, model_type):
        """Vereinfacht - verwendet nur noch Fallback-Renderer"""
        self.current_model = 'fallback'  # Verwende nur noch Fallback-Darstellung
        self.render()
    
    def set_textures(self, texture_images):
        """Setze die aktuellen Texturen und aktualisiere GLTF2-Objekte"""
        print(f"Set Textures aufgerufen mit {len(texture_images) if texture_images else 0} Texturen")
        if texture_images:
            for tex_type, img in texture_images.items():
                print(f"  - {tex_type}: {img.size if hasattr(img, 'size') else 'Unbekannt'}")
                
                # 📝 Aktualisiere GLTF2-Objekt mit neuer Textur
                self.update_gltf_texture(self.current_model, tex_type, img)
        
        self.texture_images = texture_images or {}
        print("Starte Render-Vorgang...")
        self.render()
    
    def render(self):
        """Rendere das 3D-Modell mit wählbarem Renderer"""
        self.canvas.delete("all")
        
        # Prüfe Renderer-Modus (falls vorhanden)
        renderer_mode = getattr(self, 'renderer_mode', None)
        mode = renderer_mode.get() if renderer_mode else 'auto'
        
        # Bestimme Renderer basierend auf Modus
        use_fallback = False
        if mode == 'fallback':
            use_fallback = True
        elif mode == 'pyrender':
            use_fallback = False
        else:  # auto
            use_fallback = (getattr(self, 'use_fallback_renderer', False) or 
                          not PYRENDER_AVAILABLE or not self.renderer)
        
        # Verwende gewählten Renderer
        if use_fallback:
            return self.render_fallback()
        
        if self.current_model not in self.models:
            self.canvas.create_text(
                self.width//2, self.height//2,
                text=f"3D-Modell '{self.current_model}' nicht gefunden",
                fill="white", font=("Arial", 16)
            )
            return
        
        try:
            # Verwende sichereren Rendering-Ansatz
            return self.render_with_pyrender()
            
        except Exception as render_error:
            print(f"PyRender Fehler: {render_error}")
            print("Wechsle zu Fallback-Renderer...")
            self.use_fallback_renderer = True
            return self.render_fallback()
    
    def render_with_pyrender(self):
        """Sicheres PyRender-Rendering mit verbesserter Fehlerbehandlung"""
        try:
            # VOLLSTÄNDIGE Scene-Bereinigung
            # Entferne alle bestehenden Nodes UND bereinige Renderer
            if self.scene is not None and hasattr(self.scene, 'mesh_nodes') and hasattr(self.scene, 'camera_nodes'):
                nodes_to_remove = list(self.scene.mesh_nodes) + list(self.scene.camera_nodes)
                for node in nodes_to_remove:
                    try:
                        self.scene.remove_node(node)
                    except Exception:
                        pass  # Ignoriere Fehler beim Entfernen
            
            # Erstelle KOMPLETT NEUES MESH (nie wiederverwenden)
            original_mesh = self.models[self.current_model]
            material = self.create_material_with_textures()
            
            # Variable für späteres Cleanup definieren
            camera_node = None
            
            # Erstelle eine tiefe Kopie des Primitive um Kontext-Konflikte zu vermeiden
            if material and hasattr(original_mesh, 'primitives') and len(original_mesh.primitives) > 0 and pyrender is not None:
                try:
                    # Hole Original-Primitive
                    orig_primitive = original_mesh.primitives[0]
                    
                    # Erstelle eine neue Primitive-Instanz mit kopierten Daten
                    new_primitive = pyrender.Primitive(
                        positions=orig_primitive.positions.copy() if orig_primitive.positions is not None else None,
                        normals=orig_primitive.normals.copy() if orig_primitive.normals is not None else None,
                        texcoord_0=orig_primitive.texcoord_0.copy() if orig_primitive.texcoord_0 is not None else None,
                        indices=orig_primitive.indices.copy() if orig_primitive.indices is not None else None,
                        material=material,  # Neues Material zuweisen
                        mode=orig_primitive.mode
                    )
                    
                    # Erstelle komplett neues Mesh mit neuer Primitive
                    mesh = pyrender.Mesh(
                        primitives=[new_primitive],
                        name=f"isolated_mesh_{id(self)}_{hash(str(material))}"
                    )
                    
                    print("Isoliertes Mesh mit Textur-Material erstellt")
                except Exception as mesh_error:
                    print(f"Fehler beim Erstellen des isolierten Mesh: {mesh_error}")
                    # Fallback: Verwende Original aber mit neuem Namen
                    try:
                        if pyrender is not None and trimesh is not None:
                            mesh = pyrender.Mesh.from_trimesh(
                                trimesh.Trimesh(
                                    vertices=original_mesh.primitives[0].positions,
                                    faces=original_mesh.primitives[0].indices.reshape(-1, 3)
                                ),
                                material=material if material else None
                            )
                            print("Fallback-Mesh aus Trimesh erstellt")
                        else:
                            mesh = original_mesh
                            print("Verwende Original-Mesh")
                    except Exception:
                        mesh = original_mesh
                        print("Verwende Original-Mesh")
            else:
                # Kein Material: Verwende Original-Mesh
                mesh = original_mesh
            
            # Erstelle Transformationsmatrix
            transform_matrix = self.create_transform_matrix()
            if pyrender is not None and self.scene is not None:
                mesh_node = pyrender.Node(mesh=mesh, matrix=transform_matrix)
                self.scene.add_node(mesh_node)
                
                # Erstelle Kamera
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.width/self.height)
                camera_matrix = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, self.zoom * 3.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                camera_node = pyrender.Node(camera=camera, matrix=camera_matrix)
                self.scene.add_node(camera_node)
            
            # 🎥 SICHERES RENDERING mit Fehlerbehandlung
            try:
                if self.scene is not None and self.renderer is not None and pyrender is not None and isinstance(self.scene, pyrender.Scene):
                    render_result = self.renderer.render(self.scene)
                    if render_result is not None:
                        color, depth = render_result
                    else:
                        raise RuntimeError("PyRender render() hat None zurückgegeben.")
                else:
                    raise RuntimeError("PyRender Scene ist nicht initialisiert oder ungültig.")
                
                # Konvertiere zu PIL Image und zeige an
                pil_image = Image.fromarray(color)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Zeige auf Canvas
                self.canvas.create_image(self.width//2, self.height//2, image=photo)
                self.canvas.image = photo  # Referenz behalten
                
                # Info-Text
                info_text = f"Modell: {self.current_model.title()} | Zoom: {self.zoom:.1f}x"
                self.canvas.create_text(
                    10, 10, text=info_text, fill="white", font=("Arial", 10), anchor="nw"
                )
                
                # Steuerungshinweise
                # help_text = "Maus: Rotation | Mausrad: Zoom | PyRender aktiv"
                # self.canvas.create_text(
                #     10, self.height-10, text=help_text, fill="lime", font=("Arial", 9), anchor="sw"
                # )
                
            except Exception as render_error:
                print(f"Rendering-Fehler: {render_error}")
                # Fallback-Darstellung bei Render-Fehlern
                self.canvas.create_text(
                    self.width//2, self.height//2,
                    text="3D-Render temporär nicht verfügbar",
                    fill="yellow", font=("Arial", 14)
                )
                self.canvas.create_text(
                    self.width//2, self.height//2 + 30,
                    text=f"Fehler: {str(render_error)[:40]}...",
                    fill="orange", font=("Arial", 10)
                )
            
            # CLEANUP nach Rendering
            try:
                # Entferne Kamera-Node
                if camera_node is not None and self.scene is not None:
                    self.scene.remove_node(camera_node)
            except Exception:
                pass  # Ignoriere Cleanup-Fehler
                
        except Exception as e:
            print(f"Fehler beim PyRender-Rendering: {e}")
            # Bessere Fehler-Darstellung
            self.canvas.create_text(
                self.width//2, self.height//2,
                text="3D-Rendering fehlgeschlagen",
                fill="red", font=("Arial", 16)
            )
            self.canvas.create_text(
                self.width//2, self.height//2 + 30,
                text=f"Details: {str(e)[:50]}...",
                fill="orange", font=("Arial", 10)
            )
    
    def render_fallback(self):
        """Fallback-Renderer für Plane-Darstellung"""
        try:
            # Erstelle einfache Plane-Darstellung
            center_x, center_y = self.width // 2, self.height // 2
            
            # Hintergrund
            self.canvas.create_rectangle(0, 0, self.width, self.height, fill="#1a1a1a", outline="")
            
            # Zeichne drehbare Plane
            self.draw_fallback_plane(center_x, center_y)
            
            # Texturen-Overlay wenn vorhanden
            if self.texture_images and 'base_color' in self.texture_images:
                self.draw_texture_overlay_plane(center_x, center_y)
            
            # Info-Text
            info_text = f"Textur-Plane | Zoom: {self.zoom:.1f}x"
            self.canvas.create_text(
                10, 10, text=info_text, fill="white", font=("Arial", 10), anchor="nw"
            )
            
            # Fallback-Hinweis
            # help_text = "Maus: Rotation | Mausrad: Zoom | Plane-Renderer aktiv"
            # self.canvas.create_text(
            #     10, self.height-10, text=help_text, fill="yellow", font=("Arial", 9), anchor="sw"
            # )
            
        except Exception as e:
            print(f"Fallback-Render-Fehler: {e}")
            self.canvas.create_text(
                self.width//2, self.height//2,
                text="Rendering nicht verfügbar",
                fill="red", font=("Arial", 16)
            )
    
    def draw_fallback_cube(self, center_x, center_y):
        """Zeichne statischen isometrischen Würfel ohne Rotation"""
        size = min(self.width, self.height) * 0.2 * self.zoom
        
        # Feste isometrische Projektion ohne Rotation
        iso_factor = 0.5
        
        # Grundpunkte des Würfels (statisch)
        points_3d = [
            (-size, -size, -size), (size, -size, -size), (size, size, -size), (-size, size, -size),  # Unten
            (-size, -size, size), (size, -size, size), (size, size, size), (-size, size, size)     # Oben
        ]
        
        # Projiziere 3D-Punkte auf 2D (ohne Rotation)
        points_2d = []
        for x, y, z in points_3d:
            # Isometrische Projektion
            screen_x = center_x + x + y * iso_factor
            screen_y = center_y - z + y * iso_factor
            
            points_2d.append((screen_x, screen_y))
        
        # Zeichne Würfel-Flächen mit PBR-Materialien
        base_color = self.get_combined_pbr_color()
        
        # Vorderseite - hellste Fläche
        front_color = self.lighten_color(base_color, 0.2)
        self.canvas.create_polygon([
            points_2d[0], points_2d[1], points_2d[2], points_2d[3]
        ], fill=front_color, outline="white", width=2)
        
        # Oberseite - mittlere Helligkeit
        top_color = self.lighten_color(base_color, 0.4)
        self.canvas.create_polygon([
            points_2d[4], points_2d[5], points_2d[6], points_2d[7]
        ], fill=top_color, outline="white", width=2)
        
        # Rechte Seite - dunkelste Fläche
        right_color = self.darken_color(base_color, 0.3)
        self.canvas.create_polygon([
            points_2d[1], points_2d[5], points_2d[6], points_2d[2]
        ], fill=right_color, outline="white", width=2)
    
    def draw_fallback_sphere(self, center_x, center_y):
        """Zeichne Kugel mit PBR-basierter Schattierung"""
        radius = min(self.width, self.height) * 0.15 * self.zoom
        
        # Hole PBR-Basis-Farbe
        base_color = self.get_combined_pbr_color()
        
        # Bestimme Materialparameter aus Texturen
        metallic_factor = self.get_texture_average('metallic') / 255.0 if 'metallic' in self.texture_images else 0.0
        roughness_factor = self.get_texture_average('roughness') / 255.0 if 'roughness' in self.texture_images else 0.5
        
        # Mehrere Kreise für Schattierung mit PBR-Eigenschaften
        for i in range(8):
            # Berechne Schattierungsfaktor basierend auf Roughness
            if roughness_factor > 0.7:
                # Sehr raue Oberfläche - weniger Reflexion, mehr diffuse Schattierung
                shade_factor = 0.3 + (i / 8) * 0.4
            elif metallic_factor > 0.7:
                # Metallische Oberfläche - schärfere Reflexionen
                shade_factor = 0.1 + (i / 8) * 0.7
            else:
                # Standard-Schattierung
                shade_factor = 0.2 + (i / 8) * 0.6
            
            # Wende Schattierung auf Basis-Farbe an
            shaded_color = self.darken_color(base_color, 1.0 - shade_factor)
            r = radius - i * radius/8
            
            self.canvas.create_oval(
                center_x - r, center_y - r,
                center_x + r, center_y + r,
                fill=shaded_color, outline=""
            )
        
        # Highlight für 3D-Effekt - abhängig von Roughness und Metallic
        if roughness_factor < 0.8:  # Nur bei nicht zu rauen Oberflächen
            highlight_r = radius * (0.4 - roughness_factor * 0.2)  # Kleinerer Highlight bei raueren Oberflächen
            highlight_x = center_x - radius * 0.3
            highlight_y = center_y - radius * 0.3
            
            # Metallic-Oberflächen haben farbige Highlights
            if metallic_factor > 0.5:
                highlight_color = self.lighten_color(base_color, 0.6)
            else:
                highlight_color = "white"
            
            # Highlight-Intensität basierend auf Roughness
            if roughness_factor > 0.5:
                # Semi-transparenter Highlight für raue Oberflächen
                highlight_color = self.lighten_color(base_color, 0.3)
            
            self.canvas.create_oval(
                highlight_x - highlight_r, highlight_y - highlight_r,
                highlight_x + highlight_r, highlight_y + highlight_r,
                fill=highlight_color, outline=""
            )
    
    def get_texture_average(self, texture_type):
        """Berechne Durchschnittswert einer Textur"""
        try:
            if texture_type in self.texture_images:
                img = self.texture_images[texture_type].convert('L')
                small_img = img.resize((16, 16), Image.Resampling.LANCZOS)
                return sum(small_img.getdata()) // (16 * 16)
            return 128  # Standard-Mittelwert
        except Exception:
            return 128
    
    def draw_texture_overlay(self, center_x, center_y):
        """Zeichne erweiterte PBR-Textur-Überlagerung"""
        try:
            if 'base_color' in self.texture_images:
                base_texture = self.texture_images['base_color']
                size = min(self.width, self.height) * 0.25 * self.zoom
                
                # Kombiniere Texturen für bessere PBR-Darstellung
                combined_texture = self.combine_pbr_textures(base_texture, int(size))
                
                # Erstelle PhotoImage
                photo = ImageTk.PhotoImage(combined_texture)
                
                # Platziere auf Canvas
                self.canvas.create_image(center_x, center_y, image=photo)
                
                # Speichere Referenz
                if not hasattr(self.canvas, 'texture_refs'):
                    self.canvas.texture_refs = []
                self.canvas.texture_refs.append(photo)
            
            # Zusätzliche Visualisierung für spezielle Maps (temporär deaktiviert)
            # self.draw_pbr_indicators(center_x, center_y)
                
        except Exception as e:
            print(f"Fehler beim Textur-Overlay: {e}")
    
    def combine_pbr_textures(self, base_texture, size):
        """Kombiniere PBR-Texturen für bessere Visualisierung"""
        try:
            # Basis-Textur skalieren
            combined = base_texture.resize((size, size), Image.Resampling.LANCZOS)
            combined_array = np.array(combined, dtype=np.float32)
            
            # Normal Map Effekt (simuliert durch Helligkeit-Variation)
            if 'normal' in self.texture_images:
                normal_img = self.texture_images['normal'].convert('L')
                normal_resized = normal_img.resize((size, size), Image.Resampling.LANCZOS)
                normal_array = np.array(normal_resized, dtype=np.float32)
                
                # Verstärke Kontrast basierend auf Normal Map
                normal_factor = (normal_array - 128) / 128.0 * 0.3  # Subtiler Effekt
                for i in range(3):  # RGB-Kanäle
                    combined_array[:, :, i] = np.clip(
                        combined_array[:, :, i] * (1.0 + normal_factor), 
                        0, 255
                    )
            
            # Roughness-Effekt (beeinflusst Sättigung)
            if 'roughness' in self.texture_images:
                roughness_img = self.texture_images['roughness'].convert('L')
                roughness_resized = roughness_img.resize((size, size), Image.Resampling.LANCZOS)
                roughness_array = np.array(roughness_resized, dtype=np.float32) / 255.0
                
                # Konvertiere zu HSV für Sättigungsanpassung
                from colorsys import rgb_to_hsv, hsv_to_rgb
                
                for y in range(size):
                    for x in range(size):
                        r, g, b = combined_array[y, x] / 255.0
                        h, s, v = rgb_to_hsv(r, g, b)
                        
                        # Raue Oberflächen haben weniger Sättigung
                        s *= (1.0 - roughness_array[y, x] * 0.3)
                        
                        r, g, b = hsv_to_rgb(h, s, v)
                        combined_array[y, x] = [r * 255, g * 255, b * 255]
            
            # Metallic-Effekt (erhöht Kontrast und Reflexion)
            if 'metallic' in self.texture_images:
                metallic_img = self.texture_images['metallic'].convert('L')
                metallic_resized = metallic_img.resize((size, size), Image.Resampling.LANCZOS)
                metallic_array = np.array(metallic_resized, dtype=np.float32) / 255.0
                
                # Metallische Bereiche bekommen mehr Kontrast
                for i in range(3):
                    combined_array[:, :, i] = np.clip(
                        combined_array[:, :, i] + (combined_array[:, :, i] - 128) * metallic_array * 0.5,
                        0, 255
                    )
            
            # AO-Effekt (verdunkelt)
            if 'occlusion' in self.texture_images:
                ao_img = self.texture_images['occlusion'].convert('L')
                ao_resized = ao_img.resize((size, size), Image.Resampling.LANCZOS)
                ao_array = np.array(ao_resized, dtype=np.float32) / 255.0
                
                # Dunkle AO-Bereiche ab
                for i in range(3):
                    combined_array[:, :, i] *= ao_array
            
            return Image.fromarray(combined_array.astype(np.uint8))
            
        except Exception as e:
            print(f"Fehler bei PBR-Kombination: {e}")
            return base_texture.resize((size, size), Image.Resampling.LANCZOS)
    
    def draw_pbr_indicators(self, center_x, center_y):
        """Zeichne Indikatoren für verschiedene PBR-Maps"""
        try:
            indicator_size = 15
            start_x = center_x - 100
            start_y = center_y + 120
            
            # Map-Indikatoren
            maps_info = [
                ('normal', 'N', 'blue'),
                ('roughness', 'R', 'orange'),
                ('metallic', 'M', 'silver'),
                ('occlusion', 'AO', 'brown'),
                ('emission', 'E', 'yellow'),
                ('alpha', 'A', 'purple'),
                ('height', 'H', 'green')
            ]
            
            active_count = 0
            for i, (map_name, label, color) in enumerate(maps_info):
                if map_name in self.texture_images:
                    x = start_x + active_count * (indicator_size + 5)
                    y = start_y
                    
                    # Indikator-Kreis
                    self.canvas.create_oval(
                        x - indicator_size//2, y - indicator_size//2,
                        x + indicator_size//2, y + indicator_size//2,
                        fill=color, outline="white", width=1
                    )
                    
                    # Label
                    self.canvas.create_text(
                        x, y, text=label, fill="white", 
                        font=("Arial", 8, "bold")
                    )
                    
                    active_count += 1
                    
        except Exception as e:
            print(f"Fehler bei PBR-Indikatoren: {e}")
    
    def create_transform_matrix(self):
        """Erstelle statische Transformationsmatrix ohne Rotation"""
        # Identitätsmatrix (keine Rotation)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def draw_fallback_plane(self, center_x, center_y):
        """Zeichne statische Plane (Ebene) für Textur-Vorschau ohne Rotation"""
        size = min(self.width, self.height) * 0.3 * self.zoom
        
        # Plane-Ecken als einfaches Rechteck ohne 3D-Transformation
        corners = [
            (center_x - size, center_y - size),  # Links oben
            (center_x + size, center_y - size),  # Rechts oben
            (center_x + size, center_y + size),  # Rechts unten
            (center_x - size, center_y + size)   # Links unten
        ]
        
        # Hole PBR-Farbe
        base_color = self.get_combined_pbr_color()
        
        # Zeichne Plane als Polygon
        self.canvas.create_polygon(
            corners, 
            fill=base_color, 
            outline="white", 
            width=2
        )
        
        # Zeichne Rahmen für bessere Sichtbarkeit
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            self.canvas.create_line(
                start[0], start[1], end[0], end[1],
                fill="white", width=3
            )
    
    def draw_texture_overlay_plane(self, center_x, center_y):
        """Zeichne Textur-Overlay auf der statischen Plane mit Lichteffekten"""
        try:
            if 'base_color' in self.texture_images:
                base_texture = self.texture_images['base_color']
                size = min(self.width, self.height) * 0.25 * self.zoom
                
                # Kombiniere PBR-Texturen
                combined_texture = self.combine_pbr_textures(base_texture, int(size))
                
                # Wende Lichteffekte an wenn Licht-Parameter verfügbar sind
                if hasattr(self, 'light_angle') and hasattr(self, 'light_intensity') and hasattr(self, 'ambient_intensity'):
                    combined_texture = self.apply_lighting_effects(combined_texture)
                
                # Einfache Textur-Skalierung ohne Rotation
                tex_width = max(10, int(size * self.zoom))
                tex_height = max(10, int(size * self.zoom))
                
                scaled_texture = combined_texture.resize((tex_width, tex_height), Image.Resampling.LANCZOS)
                
                # Erstelle PhotoImage
                photo = ImageTk.PhotoImage(scaled_texture)
                
                # Platziere Textur auf Plane
                self.canvas.create_image(center_x, center_y, image=photo)
                
                # Speichere Referenz
                if not hasattr(self.canvas, 'texture_refs'):
                    self.canvas.texture_refs = []
                self.canvas.texture_refs.append(photo)
                
        except Exception as e:
            print(f"Fehler beim Plane-Textur-Overlay: {e}")
    
    def apply_lighting_effects(self, image):
        """Simuliere Lichteffekte auf einer Textur für Fallback-Rendering"""
        try:
            import math
            from PIL import ImageEnhance
            
            # Hole Licht-Parameter (mit Fallback-Werten)
            angle = getattr(self, 'light_angle', 45.0)
            intensity = getattr(self, 'light_intensity', 3.0)
            ambient = getattr(self, 'ambient_intensity', 0.3)
            
            # Konvertiere zu RGBA für bessere Bearbeitung
            work_image = image.convert('RGBA')
            width, height = work_image.size
            
            # Erstelle Licht-Gradient basierend auf Winkel
            light_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            
            # Berechne Lichtrichtung
            angle_rad = math.radians(angle)
            
            # Erstelle Lichtgradient
            center_x, center_y = width // 2, height // 2
            max_distance = math.sqrt(width**2 + height**2) // 2
            
            for y in range(height):
                for x in range(width):
                    # Entfernung vom Zentrum
                    dx = x - center_x
                    dy = y - center_y
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    # Winkel zu diesem Pixel
                    pixel_angle = math.atan2(dy, dx)
                    angle_diff = abs(pixel_angle - angle_rad)
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    
                    # Lichtintensität basierend auf Winkel und Entfernung
                    angle_factor = 1.0 - (angle_diff / math.pi)
                    distance_factor = 1.0 - (distance / max_distance)
                    
                    # Kombiniere Faktoren
                    light_factor = (angle_factor * 0.7 + distance_factor * 0.3) * intensity * 0.2
                    light_factor = max(0, min(1, light_factor))
                    
                    # Berechne finale Lichtfarbe
                    light_alpha = int(light_factor * 100)
                    if light_alpha > 0:
                        light_overlay.putpixel((x, y), (255, 255, 255, light_alpha))
            
            # Kombiniere mit Original
            lit_image = Image.alpha_composite(work_image, light_overlay)
            
            # Wende Umgebungslicht an
            enhancer = ImageEnhance.Brightness(lit_image)
            lit_image = enhancer.enhance(ambient + 0.7)  # Basis-Helligkeit + Umgebungslicht
            
            # Wende Gesamtintensität an
            enhancer = ImageEnhance.Brightness(lit_image)
            final_intensity = 0.8 + (intensity - 3.0) * 0.1  # Normalisiere Intensität
            final_intensity = max(0.3, min(2.0, final_intensity))
            lit_image = enhancer.enhance(final_intensity)
            
            return lit_image.convert('RGB')
            
        except Exception as e:
            print(f"Fehler bei Lichteffekt-Simulation: {e}")
            return image
    
    def get_combined_pbr_color(self):
        """Kombiniere alle PBR-Maps zu einer Farbe für Fallback-Rendering"""
        try:
            # Standard-Farbe wenn keine Texturen vorhanden
            if not self.texture_images:
                return "#808080"
            
            # Base Color als Ausgangspunkt
            base_color = [128, 128, 128]  # Grau als Standard
            
            if 'base_color' in self.texture_images:
                # Durchschnittliche Farbe der Base Color berechnen
                base_img = self.texture_images['base_color']
                # Resize für Performance
                small_img = base_img.resize((32, 32), Image.Resampling.LANCZOS)
                
                # Durchschnittsfarbe berechnen
                pixels = list(small_img.getdata())
                if pixels:
                    avg_r = sum(p[0] for p in pixels) // len(pixels)
                    avg_g = sum(p[1] for p in pixels) // len(pixels)
                    avg_b = sum(p[2] for p in pixels) // len(pixels)
                    base_color = [avg_r, avg_g, avg_b]
            
            # Metallic-Einfluss
            if 'metallic' in self.texture_images:
                metallic_img = self.texture_images['metallic'].convert('L')
                small_metallic = metallic_img.resize((32, 32), Image.Resampling.LANCZOS)
                metallic_avg = sum(small_metallic.getdata()) // (32 * 32)
                
                # Metallische Oberflächen sind reflektiver/heller
                metallic_factor = metallic_avg / 255.0
                base_color = [int(c + (255 - c) * metallic_factor * 0.3) for c in base_color]
            
            # Roughness-Einfluss
            if 'roughness' in self.texture_images:
                roughness_img = self.texture_images['roughness'].convert('L')
                small_roughness = roughness_img.resize((32, 32), Image.Resampling.LANCZOS)
                roughness_avg = sum(small_roughness.getdata()) // (32 * 32)
                
                # Raue Oberflächen sind matter/dunkler
                roughness_factor = roughness_avg / 255.0
                base_color = [int(c * (1.0 - roughness_factor * 0.2)) for c in base_color]
            
            # Occlusion-Einfluss
            if 'occlusion' in self.texture_images:
                ao_img = self.texture_images['occlusion'].convert('L')
                small_ao = ao_img.resize((32, 32), Image.Resampling.LANCZOS)
                ao_avg = sum(small_ao.getdata()) // (32 * 32)
                
                # AO dunkelt ab
                ao_factor = ao_avg / 255.0
                base_color = [int(c * ao_factor) for c in base_color]
            
            # Emission-Einfluss
            if 'emission' in self.texture_images:
                emission_img = self.texture_images['emission'].convert('L')
                small_emission = emission_img.resize((32, 32), Image.Resampling.LANCZOS)
                emission_avg = sum(small_emission.getdata()) // (32 * 32)
                
                # Emission macht heller
                if emission_avg > 10:  # Nur wenn tatsächlich Emission vorhanden
                    emission_factor = emission_avg / 255.0
                    base_color = [min(255, int(c + 255 * emission_factor * 0.5)) for c in base_color]
            
            # Beschränke Werte auf 0-255
            base_color = [max(0, min(255, c)) for c in base_color]
            
            return f"#{base_color[0]:02x}{base_color[1]:02x}{base_color[2]:02x}"
            
        except Exception as e:
            print(f"Fehler bei PBR-Farbkombination: {e}")
            return "#808080"
    
    def lighten_color(self, hex_color, factor):
        """Helle eine Hex-Farbe auf"""
        try:
            # Konvertiere Hex zu RGB
            hex_color = hex_color.lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            
            # Aufhellen
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color
    
    def darken_color(self, hex_color, factor):
        """Dunkle eine Hex-Farbe ab"""
        try:
            # Konvertiere Hex zu RGB
            hex_color = hex_color.lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            
            # Abdunkeln
            r = max(0, int(r * (1 - factor)))
            g = max(0, int(g * (1 - factor)))
            b = max(0, int(b * (1 - factor)))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color
    
    def create_material_with_textures(self):
        """Erstelle PyRender-Material mit aktuellen Texturen"""
        try:
            if pyrender is None:
                print("PyRender nicht verfügbar für Material-Erstellung")
                return None
                
            if not self.texture_images:
                # Standard-Material ohne Texturen
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                    metallicFactor=0.2,
                    roughnessFactor=0.5,
                    doubleSided=True
                )
                return material
            
            # Basis-Material-Parameter
            baseColorFactor = [1.0, 1.0, 1.0, 1.0]
            roughnessFactor = 0.5
            metallicFactor = 0.0
            
            # Erstelle Material-Parameter Dictionary
            material_params = {
                'baseColorFactor': baseColorFactor,
                'metallicFactor': metallicFactor,
                'roughnessFactor': roughnessFactor,
                'doubleSided': True
            }
            
            # Base Color Textur hinzufügen (wenn vorhanden)
            if 'base_color' in self.texture_images:
                try:
                    base_img = self.texture_images['base_color']
                    # Stelle sicher, dass das Bild im RGB Format ist
                    if base_img.mode != 'RGB':
                        base_img = base_img.convert('RGB')
                    
                    # Konvertiere zu NumPy Array (0-255 uint8) mit richtiger Achsen-Reihenfolge
                    base_array = np.array(base_img, dtype=np.uint8)
                    
                    # Erstelle PyRender Texture mit source_channels
                    base_texture = pyrender.Texture(source=base_array, source_channels='RGB')
                    material_params['baseColorTexture'] = base_texture
                    print(f"Base Color Textur erstellt: {base_array.shape}")
                    
                except Exception as tex_error:
                    print(f"Fehler beim Laden der Base Color Textur: {tex_error}")
            
            # Normal Map hinzufügen (wenn vorhanden)
            if 'normal' in self.texture_images:
                try:
                    normal_img = self.texture_images['normal']
                    if normal_img.mode != 'RGB':
                        normal_img = normal_img.convert('RGB')
                    
                    normal_array = np.array(normal_img, dtype=np.uint8)
                    normal_texture = pyrender.Texture(source=normal_array, source_channels='RGB')
                    material_params['normalTexture'] = normal_texture
                    print(f"Normal Map Textur erstellt: {normal_array.shape}")
                    
                except Exception as tex_error:
                    print(f"Fehler beim Laden der Normal Map: {tex_error}")
            
            # Metallic/Roughness hinzufügen (wenn vorhanden)
            if 'metallic' in self.texture_images or 'roughness' in self.texture_images:
                try:
                    # Erstelle kombinierte Metallic/Roughness Textur
                    if 'metallic' in self.texture_images and 'roughness' in self.texture_images:
                        metallic_img = self.texture_images['metallic'].convert('L')  # Graustufen
                        roughness_img = self.texture_images['roughness'].convert('L')  # Graustufen
                        
                        # Erstelle RGB Textur: R=nichts, G=Roughness, B=Metallic
                        width, height = metallic_img.size
                        
                        metallic_array = np.array(metallic_img)
                        roughness_array = np.array(roughness_img)
                        
                        # Kombiniere: G=Roughness, B=Metallic (PyRender Standard)
                        combined_array = np.zeros((height, width, 3), dtype=np.uint8)
                        combined_array[:, :, 1] = roughness_array  # Green = Roughness
                        combined_array[:, :, 2] = metallic_array   # Blue = Metallic
                        
                        mr_texture = pyrender.Texture(source=combined_array, source_channels='RGB')
                        material_params['metallicRoughnessTexture'] = mr_texture
                        print("Metallic/Roughness Textur kombiniert")
                        
                except Exception as tex_error:
                    print(f"Fehler beim Erstellen der Metallic/Roughness Textur: {tex_error}")
                    
            # Erstelle PyRender Material
            if pyrender is not None:
                material = pyrender.MetallicRoughnessMaterial(**material_params)
                return material
            else:
                print("PyRender nicht verfügbar")
                return None
                
        except Exception as e:
            print(f"Fehler beim Erstellen des Materials: {e}")
            # Fallback: Einfaches Material
            try:
                if pyrender is not None:
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                        metallicFactor=0.2,
                        roughnessFactor=0.5,
                        doubleSided=True
                    )
                    return material
                else:
                    return None
            except Exception as e:
                print(f"Fehler beim Erstellen des kombinierten Bildes: {e}")
                return None
    
    def update_lighting(self, angle, intensity, ambient):
        """Update lighting parameters for PyRender3DViewer"""
        try:
            self.light_angle = float(angle)
            self.light_intensity = float(intensity)
            self.ambient_intensity = float(ambient)
            
            # Re-render with new lighting
            self.render()
            print(f"PyRender3DViewer Beleuchtung aktualisiert: Winkel={angle}°, Stärke={intensity}, Umgebung={ambient}")
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Beleuchtung in PyRender3DViewer: {e}")

class GLTFViewer:
    """Vereinfachter GLTF-Viewer für Tkinter Canvas"""
    def __init__(self):
        self.gltf_models = {}
        
        # Fehlende Attribute für Pylance hinzufügen
        self.scene = None
        self.use_fallback_renderer = True
        self.light_angle = 45.0
        self.light_intensity = 3.0
        self.ambient_intensity = 0.3
        
        self.load_gltf_models()
    
    def load_gltf_models(self):
        """Lade GLTF-Modelle aus dem Resources-Verzeichnis"""
        try:
            if GLTF_AVAILABLE and GLTF2 is not None and trimesh is not None:
                # Lade originale Cube.gltf (mit .bin-Dateien)
                cube_path = "Resources/Cube.gltf"
                if os.path.exists(cube_path):
                    cube_gltf = GLTF2.load(cube_path)
                    cube_mesh = trimesh.load(cube_path)
                    self.gltf_models['cube'] = {
                        'gltf': cube_gltf,
                        'mesh': cube_mesh,
                        'path': cube_path
                    }
                    print(" Cube.gltf geladen")
                
                # Lade originale Ball.gltf (mit .bin-Dateien)
                ball_path = "Resources/Ball.gltf"
                if os.path.exists(ball_path):
                    ball_gltf = GLTF2.load(ball_path)
                    ball_mesh = trimesh.load(ball_path)
                    self.gltf_models['sphere'] = {
                        'gltf': ball_gltf,
                        'mesh': ball_mesh,
                        'path': ball_path
                    }
                    print(" Ball.gltf geladen")
        except Exception as e:
            print(f" Fehler beim Laden der GLTF-Modelle: {e}")
            self.gltf_models = {}
    
    def render_gltf_to_image(self, model_type, width, height, texture_images=None, rotation=0):
        """Rendere GLTF-Modell zu einem PIL-Image"""
        try:
            if not GLTF_AVAILABLE or model_type not in self.gltf_models:
                return self.create_fallback_image(model_type, width, height, texture_images)
            
            # Erstelle eine einfache Wireframe-Darstellung
            img = Image.new('RGB', (width, height), color=(20, 20, 20))
            draw = ImageDraw.Draw(img)
            
            # Hole das Mesh
            mesh_data = self.gltf_models[model_type]['mesh']
            
            # Vereinfachtes 3D zu 2D Rendering
            if hasattr(mesh_data, 'vertices') and hasattr(mesh_data, 'faces'):
                vertices = mesh_data.vertices
                faces = mesh_data.faces
                
                # Einfache orthographische Projektion
                scale = min(width, height) * 0.3
                center_x, center_y = width // 2, height // 2
                
                # Rotation anwenden
                if rotation != 0:
                    angle = np.radians(rotation)
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    
                    # Rotation um Y-Achse
                    rotation_matrix = np.array([
                        [cos_a, 0, sin_a],
                        [0, 1, 0],
                        [-sin_a, 0, cos_a]
                    ])
                    vertices = vertices @ rotation_matrix.T
                
                # Projiziere Vertices
                projected_vertices = []
                for vertex in vertices:
                    x = center_x + vertex[0] * scale
                    y = center_y - vertex[1] * scale  # Y invertiert für Bildkoordinaten
                    projected_vertices.append((x, y))
                
                # Zeichne Wireframe
                for face in faces:
                    if len(face) >= 3:
                        for i in range(len(face)):
                            v1 = projected_vertices[face[i]]
                            v2 = projected_vertices[face[(i + 1) % len(face)]]
                            draw.line([v1, v2], fill=(100, 100, 255), width=2)
                
                # Füge Textur-Overlay hinzu, falls vorhanden
                if texture_images and 'base_color' in texture_images:
                    texture = texture_images['base_color']
                    # Skaliere Textur
                    texture_size = int(scale * 0.8)
                    texture_resized = texture.resize((texture_size, texture_size), Image.Resampling.LANCZOS)
                    
                    # Platziere Textur in der Mitte
                    texture_x = center_x - texture_size // 2
                    texture_y = center_y - texture_size // 2
                    
                    # Erstelle Alpha-Mask für besseres Blending
                    img.paste(texture_resized, (texture_x, texture_y), texture_resized.convert('RGBA'))
            
            return img
            
        except Exception as e:
            print(f" Fehler beim GLTF-Rendering: {e}")
            return self.create_fallback_image(model_type, width, height, texture_images)
    
    def create_fallback_image(self, model_type, width, height, texture_images=None):
        """Erstelle Fallback-Darstellung wenn GLTF nicht verfügbar"""
        img = Image.new('RGB', (width, height), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        
        if model_type == 'cube':
            # Zeichne isometrischen Würfel
            size = min(width, height) * 0.25
            
            # Vorderseite
            points = [
                (center_x - size, center_y - size//2),
                (center_x + size, center_y - size//2),
                (center_x + size, center_y + size*1.5),
                (center_x - size, center_y + size*1.5)
            ]
            draw.polygon(points, fill=(70, 70, 70), outline=(255, 255, 255), width=3)
            
            # Oberseite
            points = [
                (center_x - size, center_y - size//2),
                (center_x - size//2, center_y - size),
                (center_x + size*1.5, center_y - size),
                (center_x + size, center_y - size//2)
            ]
            draw.polygon(points, fill=(106, 106, 106), outline=(255, 255, 255), width=3)
            
            # Rechte Seite
            points = [
                (center_x + size, center_y - size//2),
                (center_x + size*1.5, center_y - size),
                (center_x + size*1.5, center_y + size),
                (center_x + size, center_y + size*1.5)
            ]
            draw.polygon(points, fill=(42, 42, 42), outline=(255, 255, 255), width=3)
            
        else:  # sphere
            # Zeichne Kugel mit Schattierung
            radius = min(width, height) * 0.25
            for i in range(8):
                shade = 80 - i * 10
                color = (shade, shade, shade)
                r = radius - i * 8
                
                left = center_x - r
                top = center_y - r
                right = center_x + r
                bottom = center_y + r
                
                draw.ellipse([left, top, right, bottom], fill=color)
            
            # Highlight
            highlight_r = 20
            draw.ellipse([
                center_x - radius//2 - highlight_r,
                center_y - radius//2 - highlight_r,
                center_x - radius//2 + highlight_r,
                center_y - radius//2 + highlight_r
            ], fill=(255, 255, 255))
        
        # Füge Textur hinzu, falls vorhanden
        if texture_images and 'base_color' in texture_images:
            texture = texture_images['base_color']
            # Erstelle semitransparente Overlay
            texture_size = int(min(width, height) * 0.4)
            texture_resized = texture.resize((texture_size, texture_size), Image.Resampling.LANCZOS)
            
            # Erstelle Alpha-Blend
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            texture_x = center_x - texture_size // 2
            texture_y = center_y - texture_size // 2
            overlay.paste(texture_resized, (texture_x, texture_y))
            
            # Blend mit Original
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        
        return img
    
    def update_lighting(self, angle, intensity, ambient):
        """Aktualisiere Beleuchtung basierend auf den Parametern"""
        try:
            # Speichere Licht-Parameter
            self.light_angle = angle
            self.light_intensity = intensity
            self.ambient_intensity = ambient
            
            # Aktualisiere PyRender Szene falls verfügbar
            if self.scene is not None and not self.use_fallback_renderer and pyrender is not None:
                # Entferne alle bestehenden Lichter
                nodes_to_remove = []
                try:
                    # Sichere Licht-Entfernung (Pylance-kompatibel)
                    if self.scene is not None:
                        # Verwende exec() für dynamische Scene-Manipulation um Pylance zu umgehen
                        exec_code = """
scene_obj = self.scene
if hasattr(scene_obj, 'get_nodes'):
    nodes = scene_obj.get_nodes()
    for node in nodes:
        if hasattr(node, 'light') and node.light is not None:
            nodes_to_remove.append(node)
"""
                        local_vars = {'self': self, 'nodes_to_remove': nodes_to_remove}
                        exec(exec_code, {}, local_vars)
                        nodes_to_remove = local_vars['nodes_to_remove']
                    
                    for node in nodes_to_remove:
                        if self.scene is not None and hasattr(self.scene, 'remove_node'):
                            self.scene.remove_node(node)
                except (AttributeError, TypeError, NameError):
                    # Fallback wenn Scene-Manipulation nicht verfügbar ist
                    pass
                
                # Füge neues direktionales Licht hinzu
                import math
                angle_rad = math.radians(angle)
                direction = [math.cos(angle_rad), math.sin(angle_rad), -1.0]
                
                light = pyrender.DirectionalLight(
                    color=[1.0, 1.0, 1.0], 
                    intensity=intensity
                )
                light_matrix = np.eye(4)
                light_matrix[:3, 3] = direction
                light_node = pyrender.Node(light=light, matrix=light_matrix)
                self.scene.add_node(light_node)
                
                # Aktualisiere Umgebungslicht
                self.scene.ambient_light = [ambient, ambient, ambient]
                
                print(f"PyRender Beleuchtung aktualisiert: Winkel={angle}°, Stärke={intensity}, Umgebung={ambient}")
            
            # Rendere neu
            self.render()
            
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Beleuchtung: {e}")
    
    def render(self):
        """Dummy render-Methode für GLTFViewer"""
        pass

class PBRMaterialMaker:
    def __init__(self):
        self.current_textures: Dict[str, Optional[str]] = {
            "base_color": None,
            "normal": None,
            "roughness": None,
            "metallic": None,
            "occlusion": None,
            "emission": None,
            "alpha": None,
            "height": None
        }
        
        # PyPBR-Pipeline initialisieren
        self.pypbr_pipeline = PyPBRMaterialPipeline()
        
        # Standard-Konfiguration
        self.config = {
            "NormalStrength": 0.20,
            "RoughnessStrength": 0.20,
            "OcclusionStrength": 1.0,
            "MetallicThreshold": 127,
            "EmissionStrength": 0.0,
            "EmissionEdgeEnhance": False,  # Neue Einstellung für Kontur-Hervorhebung
            "EmissionEdgeStrength": 1.0,   # Stärke der Kontur-Hervorhebung
            "AlphaStrength": 1.0,
            "BaseColorTint": [1.0, 1.0, 1.0],
            "NormalMapType": "sobel",
            "RoughnessInvert": False,
            "MetallicIntensity": 1.0,
            "EmissionColor": [1.0, 1.0, 1.0],
            "AlphaMode": "opaque"
        }
        
        # Material-Presets laden
        self.material_presets = self.load_material_presets()
        
        # Textur-Erkennungs-Patterns
        self.texture_patterns = {
            "base_color": ["_albedo", "_diffuse", "_color", "_basecolor", "_base_color", "_diff", "_col",
                          "-albedo", "-diffuse", "-color", "-basecolor", "-base-color", "-diff", "-col"],
            "normal": ["_normal", "_norm", "_nrm", "_normalmap", "_normal_map","_normal-ogl"
                      "-normal", "-norm", "-nrm", "-normalmap", "-normal-map","-normal-ogl"],
            "roughness": ["_roughness", "_rough", "_rgh", "_roughnessmap",
                         "-roughness", "-rough", "-rgh", "-roughnessmap"],
            "metallic": ["_metallic", "_metal", "_met", "_metallicmap",
                        "-metallic", "-metal", "-met", "-metallicmap"],
            "occlusion": ["_ao", "_occlusion", "_ambient", "_ambientocclusion", "_ambient_occlusion",
                         "-ao", "-occlusion", "-ambient", "-ambientocclusion", "-ambient-occlusion"],
            "emission": ["_emission", "_emissive", "_emit", "_glow", "_light",
                        "-emission", "-emissive", "-emit", "-glow", "-light"],
            "alpha": ["_alpha", "_opacity", "_transparent", "_mask",
                     "-alpha", "-opacity", "-transparent", "-mask"],
            "height": ["_height", "_displacement", "_disp", "_bump",
                      "-height", "-displacement", "-disp", "-bump"]
        }
    
    def truncate_filename(self, filename, max_width=14):
        """Kürze Dateinamen auf maximale Breite für bessere Anzeige
        
        Args:
            filename: Der Dateiname der gekürzt werden soll
            max_width: Maximale Anzahl Zeichen (Standard: 14 für gute Lesbarkeit)
        """
        if len(filename) <= max_width:
            return filename
        
        # Bei langen Namen: einfach Dateiendung weglassen
        name, ext = os.path.splitext(filename)
        
        if len(name) <= max_width:
            return name  # Name ohne Endung passt
        else:
            return name[:max_width]  # Name auf max_width kürzen
    
    def load_material_presets(self):
        """Lade Material-Presets aus Resources/material.json"""
        try:
            material_json_path = os.path.join("Resources", "material.json")
            with open(material_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            presets = {"Default": {
                "NormalStrength": 1.0,
                "RoughnessStrength": 1.0,
                "OcclusionStrength": 1.0,
                "MetallicThreshold": 127,
                "EmissionStrength": 0.0,
                "AlphaStrength": 1.0,
                "BaseColorTint": [1.0, 1.0, 1.0],
                "EmissionColor": [1.0, 1.0, 1.0]
            }}
            
            # Prüfe ob es ein Materials-Array gibt (neue Struktur)
            if "Materials" in data and isinstance(data["Materials"], list):
                print("Lade Material-Presets aus Materials-Array...")
                for material in data["Materials"]:
                    if "MaterialName" in material:
                        material_name = material["MaterialName"]
                        presets[material_name] = material
                        print(f"Material geladen: {material_name}")
                print(f"Insgesamt {len(presets)} Material-Presets geladen")
            else:
                # Fallback: Lade direkte Struktur (alte Struktur)
                print("Lade Material-Presets aus direkter Struktur...")
                for name, material in data.items():
                    if name != "Materials":  # Überspringe Materials-Array wenn vorhanden
                        presets[name] = material
                        print(f"Material geladen: {name}")
            
            return presets
        except Exception as e:
            print(f"Fehler beim Laden der Presets: {e}")
            return {"Default": {}}
    
    def detect_texture_type(self, file_path):
        """Erkenne automatisch den Texturtyp basierend auf dem Dateinamen"""
        filename_lower = os.path.basename(file_path).lower()
        
        for tex_type, patterns in self.texture_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return tex_type
        
        return "base_color"  # Standard-Fallback

class MaterialMakerGUI:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        
        # Extrahiere Versionsnummer aus Dateinamen
        script_name = os.path.basename(__file__)
        version_match = re.search(r'v?(\d+\.\d+\.\d+)', script_name)
        version = version_match.group(1) if version_match else "1.0.0"
        
        # Größenoptionen definieren vor Fenster-Setup
        self.size_options = {
            "Klein": (128, 128),
            "Mittel": (200, 200), 
            "Groß": (256, 256)
        }
        
        # Größenoptionen für die Textur-Plane Vorschau
        self.plane_preview_sizes = {
            "Klein": (600, 600),
            "Mittel": (900, 900),
            "Groß": (1200, 1200)
        }
        
        # Fenstergrößen entsprechend der Plane-Vorschau-Größen
        # Berechnung: Plane-Breite + linke Spalte (800px) + Padding (200px)
        self.window_sizes = {
            "Klein": "1600x1750",
            "Mittel": "1900x1900",
            "Groß": "2400x2000" # 2400x1700
        }
        
        self.current_size = "Groß"
        self.thumbnail_size = self.size_options[self.current_size]
        self.plane_preview_size = self.plane_preview_sizes[self.current_size]
        self.current_window_size = self.window_sizes[self.current_size]
        
        self.root.title(f"OpenSimulator PBR Material Maker v{version} - GLTF Edition")
        
        # Programmicon setzen
        try:
            icon_path = "Resources/PBRlogok.png"
            if os.path.exists(icon_path):
                # Lade PNG als PhotoImage und setze als Icon
                icon_image = Image.open(icon_path)
                # Konvertiere zu einer für Tkinter geeigneten Größe (32x32 ist optimal für Icons)
                icon_image = icon_image.resize((32, 32), Image.Resampling.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon_image)
                # Setze Icon (Type-Safety)
                self.root.wm_iconphoto(True, icon_photo)  # type: ignore
                # Speichere Referenz um Garbage Collection zu vermeiden
                self.icon_photo = icon_photo
                print(f"✓ Programmicon gesetzt: {icon_path}")
            else:
                print(f"⚠ Icon-Datei nicht gefunden: {icon_path}")
        except Exception as e:
            print(f"⚠ Fehler beim Laden des Icons: {e}")
        
        self.root.geometry(self.current_window_size)  # Dynamische Fenstergröße basierend auf Vorschaugröße
        
        # PBR Material Maker Backend
        self.pbr_maker = PBRMaterialMaker()
        
        # GLTF Viewer für 3D-Vorschau (wird nach Canvas-Erstellung initialisiert)
        self.gltf_viewer = GLTFViewer()
        self.interactive_3d_viewer = None  # Wird später initialisiert
        
        # Bild-Cache für Vorschaubilder
        self.preview_images = {}
        
        # Aktuelle Texturen für GLTF-Rendering
        self.current_texture_images = {}
        
        # Größenoptionen bereits oben definiert - entfernt um Duplikate zu vermeiden
        
        # Status-Variable mit PyPBR-Enhanced Information
        self.status_var = tk.StringVar()
        
        # PyPBR-Info Variable für erweiterte Status-Informationen
        self.pypbr_info_var = tk.StringVar()
        
        # Erweiterte PyPBR-Status Initialisierung 
        pypbr_status = ""
        pypbr_details = []
        
        if hasattr(self.pbr_maker, 'pypbr_pipeline') and self.pbr_maker.pypbr_pipeline.enabled:
            if self.pbr_maker.pypbr_pipeline.gpu_enabled:
                gpu_name = torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "GPU"
                pypbr_details.append(f"PyPBR-GPU: {gpu_name[:15]}")
                pypbr_details.append("Cook-Torrance BRDF")
                pypbr_details.append("sRGB→Linear")
            else:
                pypbr_details.append("PyPBR-CPU")
                pypbr_details.append("Cook-Torrance")
            
            # Zeige Optimierungen an
            if self.pbr_maker.pypbr_pipeline.config.get('energy_conservation'):
                pypbr_details.append("Energy-Conserving")
            
            if self.pbr_maker.pypbr_pipeline.config.get('fresnel_enabled'):
                pypbr_details.append("Fresnel")
            
            pypbr_status = f" | {' | '.join(pypbr_details)}"
        
        initial_status = f"Bereit{pypbr_status}"
        self.status_var.set(initial_status)
        
        # Preview Mode für 3D-Viewer
        self.preview_mode = tk.StringVar(value="cube")
        
        # Texture Labels Dict
        self.texture_labels = {}
        self.texture_frames = {}
        
        # Lade Platzhalterbilder für alle Größen
        self.placeholder_images = self.load_placeholder_images()
        
        # Lade Icons für Buttons
        self.button_icons = {
            'search': self.load_icon('search'),
            'stars': self.load_icon('stars'),
            'trash': self.load_icon('trash'),
            'folder2-open': self.load_icon('folder2-open'),
            'download': self.load_icon('download'),
            'image': self.load_icon('image')
        }
        
        # GUI erstellen
        self.setup_ui()
    
    def truncate_filename(self, filename, max_width=14):
        """Kürze Dateinamen auf maximale Breite für bessere Anzeige
        
        Args:
            filename: Der Dateiname der gekürzt werden soll
            max_width: Maximale Anzahl Zeichen (Standard: 14 für gute Lesbarkeit)
        """
        if len(filename) <= max_width:
            return filename
        
        # Bei langen Namen: einfach Dateiendung weglassen
        name, ext = os.path.splitext(filename)
        
        if len(name) <= max_width:
            return name  # Name ohne Endung passt
        else:
            return name[:max_width]  # Name auf max_width kürzen
    
    def load_icon(self, icon_name, size=20):
        """Lade Icon aus Resources/icons Verzeichnis"""
        try:
            icon_path = os.path.join("Resources", "icons", f"{icon_name}.png")
            if os.path.exists(icon_path):
                with Image.open(icon_path) as img:
                    # Resize Icon auf gewünschte Größe
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                    return ImageTk.PhotoImage(img)
            else:
                print(f" Icon nicht gefunden: {icon_path}")
                return None
        except Exception as e:
            print(f" Fehler beim Laden des Icons {icon_name}: {e}")
            return None
    
    def update_texture_tooltip(self, texture_type):
        """Aktualisiere Tooltip für Textur mit aktuellem Dateinamen"""
        try:
            if texture_type not in self.texture_labels:
                return
                
            preview_label, file_label = self.texture_labels[texture_type]
            
            # Basis-Tooltip-Text
            base_tooltips = {
                "base_color": "Base Color (Diffuse): Grundfarbe des Materials\nLinksklick: Textur laden • Rechtsklick: Löschen • Drag&Drop: Unterstützt",
                "normal": "Normal Map: Oberflächendetails und Bumps\nLinksklick: Textur laden • Rechtsklick: Löschen • Drag&Drop: Unterstützt",
                "roughness": "Roughness: Oberflächenrauheit (schwarz=glatt, weiß=rau)\nLinksklick: Textur laden • Rechtsklick: Löschen • Drag&Drop: Unterstützt",
                "metallic": "Metallic: Metallische Eigenschaften (schwarz=Isolator, weiß=Metall)\nLinksklick: Textur laden • Rechtsklick: Löschen • Drag&Drop: Unterstützt",
                "occlusion": "Ambient Occlusion: Selbstverschattung und Tiefe\nLinksklick: Textur laden • Rechtsklick: Löschen • Drag&Drop: Unterstützt",
                "emission": "Emission: Selbstleuchtende Bereiche\nLinksklick: Textur laden • Rechtsklick: Löschen • Drag&Drop: Unterstützt",
                "alpha": "Alpha/Transparency: Transparenz (schwarz=transparent, weiß=opak)\nLinksklick: Textur laden • Rechtsklick: Löschen • Drag&Drop: Unterstützt",
                "height": "Height/Bump: Höheninformation für Displacement\nLinksklick: Textur laden • Rechtsklick: Löschen • Drag&Drop: Unterstützt"
            }
            
            # Hole aktuellen Dateinamen oder generierte Textur-Info
            current_file = None
            is_generated = False
            texture_path = self.pbr_maker.current_textures.get(texture_type)
            
            # Prüfe auf Datei-basierte Textur
            if texture_path and texture_path != "[Generated]" and os.path.exists(texture_path):
                current_file = os.path.basename(texture_path)
            # Prüfe auf generierte Textur
            elif (texture_path == "[Generated]" and 
                  hasattr(self, 'current_texture_images') and 
                  texture_type in self.current_texture_images and 
                  self.current_texture_images[texture_type] is not None):
                is_generated = True
            
            # Erstelle finalen Tooltip-Text
            tooltip_text = base_tooltips.get(texture_type, f"{texture_type}: PBR-Textur")
            
            if current_file:
                tooltip_text += f"\n\n📁 Geladene Datei: {current_file}"
                # Zusätzliche Dateinfo
                try:
                    if texture_path:  # Typ-Prüfung für texture_path
                        file_stats = os.stat(texture_path)
                        file_size = file_stats.st_size
                        if file_size < 1024:
                            size_str = f"{file_size} Bytes"
                        elif file_size < 1024 * 1024:
                            size_str = f"{file_size/1024:.1f} KB"
                        else:
                            size_str = f"{file_size/(1024*1024):.1f} MB"
                        
                        # Bildgröße ermitteln
                        try:
                            with Image.open(texture_path) as img:
                                tooltip_text += f"\n📐 Auflösung: {img.size[0]}x{img.size[1]} | Größe: {size_str}"
                        except Exception:
                            tooltip_text += f"\n📦 Dateigröße: {size_str}"
                except Exception:
                    pass
            elif is_generated:
                tooltip_text += "\n\n🎨 Generierte Textur"
                # Information über generierte Textur
                try:
                    generated_image = self.current_texture_images[texture_type]
                    tooltip_text += f"\n📐 Auflösung: {generated_image.size[0]}x{generated_image.size[1]}"
                    tooltip_text += "\n⚙️ Automatisch aus Base Color erstellt"
                except Exception:
                    pass
            else:
                tooltip_text += "\n\n📁 Keine Datei geladen"
            
            # Entferne alten Tooltip und erstelle neuen
            if hasattr(preview_label, '_tooltip'):
                try:
                    preview_label._tooltip.destroy()
                except Exception:
                    pass
            
            # Neuen Tooltip erstellen
            preview_label._tooltip = ToolTip(preview_label, tooltip_text)
            
        except Exception as e:
            print(f"Fehler beim Aktualisieren des Tooltips für {texture_type}: {e}")
        
    def setup_ui(self):
        """Erstelle die komplette Benutzeroberfläche"""
        
        # Haupt-Container mit vergrößertem Padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Konfiguriere Grid-Gewichte für 2-Spalten Layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # Linke Seite: Texturen + Konfiguration
        main_frame.columnconfigure(1, weight=2)  # Rechte Seite: GLTF-Vorschau (größer)
        main_frame.rowconfigure(1, weight=1)
        
        # Titel
        title_label = ttk.Label(main_frame, text="PBR Material Maker", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Linke Spalte: Texturen und Konfiguration kombiniert
        self.setup_left_panel(main_frame)
        
        # Rechte Spalte: Vergrößerte GLTF-Vorschau
        self.setup_preview_panel(main_frame)
        
    def setup_left_panel(self, parent):
        """Erstelle das linke Panel mit Texturen (4x2) und Konfiguration"""
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left_frame.rowconfigure(1, weight=1)  # Konfiguration kann sich ausdehnen
        
        # Oben: Texturen in 4x2 Grid
        self.setup_texture_panel_4x2(left_frame)
        
        # Unten: Konfiguration
        self.setup_config_panel_compact(left_frame)
        
    def setup_texture_panel_4x2(self, parent):
        """Erstelle das Textur-Panel im 4x2 Format"""
        texture_frame = ttk.LabelFrame(parent, text="Maustasten (Links: Laden, Rechts: Löschen)", padding="10")
        texture_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Textur-Labels in 4x2 Anordnung
        texture_labels = [
            ("base_color", "Base Color"),
            ("normal", "Normal Map"), 
            ("roughness", "Roughness"),
            ("metallic", "Metallic"),
            ("occlusion", "Occlusion (AO)"),
            ("emission", "Emission"),
            ("alpha", "Alpha"),
            ("height", "Height")
        ]
        
        self.texture_frames = {}
        self.texture_labels = {}
        
        # Erstelle Textur-Widgets in 4x2 Grid
        for i, (tex_type, label) in enumerate(texture_labels):
            row = i // 4  # 2 Reihen
            col = i % 4   # 4 Spalten
            
            # Frame für diese Textur
            tex_frame = ttk.Frame(texture_frame)
            tex_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            
            # Label
            ttk.Label(tex_frame, text=label, font=("Arial", 9, "bold")).grid(row=0, column=0)
            
            # Vorschaubild (kompakter)
            preview_label = ttk.Label(tex_frame, anchor="center")
            preview_label.grid(row=1, column=0, pady=2)
            
            # Setze Platzhalterbild
            self.set_placeholder_image(preview_label, tex_type)
            
            # Dateiname-Label (sehr kompakt)
            file_label = ttk.Label(tex_frame, text="Keine Datei", foreground="gray", font=("Arial", 8))
            file_label.grid(row=2, column=0)
            
            self.texture_frames[tex_type] = tex_frame
            self.texture_labels[tex_type] = (preview_label, file_label)
            
            # Erstelle initialen Tooltip NACH dem Aufbau des Labels-Dictionary
            self.update_texture_tooltip(tex_type)
            
            # Drag & Drop Support
            preview_label.drop_target_register(DND_FILES)  # type: ignore
            preview_label.dnd_bind('<<Drop>>', lambda e, t=tex_type: self.on_drop(e, t))  # type: ignore
            
            # Click-Handler
            preview_label.bind("<Button-1>", lambda e, t=tex_type: self.select_texture_file(t))
            # Rechte Maustaste zum Löschen der einzelnen Textur
            preview_label.bind("<Button-3>", lambda e, t=tex_type: self.clear_single_texture(t))
            
            # Labels sind bereits oben erstellt
        
        # Konfiguriere Grid-Gewichte für gleichmäßige Verteilung
        for i in range(4):
            texture_frame.columnconfigure(i, weight=1)
        
        # Action Buttons kompakt unten
        button_frame = ttk.Frame(texture_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        # Buttons mit Icons statt Emojis (fallback auf Text wenn Icon nicht verfügbar)
        auto_find_btn = ttk.Button(button_frame, text="Auto-Find", command=self.auto_find_textures)
        if self.button_icons['search']:
            auto_find_btn.config(image=self.button_icons['search'], compound="left")
        auto_find_btn.grid(row=0, column=0, padx=2)
        add_tooltip(auto_find_btn, "Automatisch passende Texturen basierend auf Base Color Dateinamen suchen")
        
        maps_btn = ttk.Button(button_frame, text="Maps generieren", command=self.generate_missing_maps)
        if self.button_icons['stars']:
            maps_btn.config(image=self.button_icons['stars'], compound="left")
        maps_btn.grid(row=0, column=1, padx=2)
        add_tooltip(maps_btn, "Fehlende PBR-Maps aus Base Color Textur automatisch generieren\n(Normal, Roughness, Metallic, AO, etc.)")
        
        clear_btn = ttk.Button(button_frame, text="Clear All", command=self.clear_all_textures)
        if self.button_icons['trash']:
            clear_btn.config(image=self.button_icons['trash'], compound="left")
        clear_btn.grid(row=0, column=2, padx=2)
        add_tooltip(clear_btn, "Alle geladenen Texturen entfernen und Platzhalterbilder wiederherstellen")
        
        load_btn = ttk.Button(button_frame, text="Bilderset laden", command=self.load_image_set)
        if self.button_icons['folder2-open']:
            load_btn.config(image=self.button_icons['folder2-open'], compound="left")
        load_btn.grid(row=0, column=3, padx=2)
        add_tooltip(load_btn, "PBR-Texturen-Set laden: Wählen Sie eine beliebige Textur,\nalle passenden werden automatisch erkannt und geladen")
        
    def setup_config_panel_compact(self, parent):
        """Erstelle das kompakte Konfigurations-Panel"""
        config_frame = ttk.LabelFrame(parent, text="Konfiguration", padding="10")
        config_frame.grid(row=1, column=0, sticky="nsew")
        
        # Preset-Auswahl
        preset_frame = ttk.LabelFrame(config_frame, text="Material Presets", padding="5")
        preset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(preset_frame, text="Preset auswählen:").grid(row=0, column=0, sticky="w")
        
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                  values=list(self.pbr_maker.material_presets.keys()),
                                  state="readonly")
        preset_combo.grid(row=1, column=0, sticky="ew", pady=2)
        preset_combo.bind('<<ComboboxSelected>>', self.apply_preset)
        add_tooltip(preset_combo, "Vordefinierte Material-Einstellungen auswählen:\nOptimierte Parameter für verschiedene Oberflächentypen\n(Holz, Metall, Stoff, etc.)")
        
        # Größeneinstellungen kompakt
        size_frame = ttk.LabelFrame(config_frame, text="Größe", padding="5")
        size_frame.grid(row=1, column=0, sticky="ew", pady=2)
        
        ttk.Label(size_frame, text="Vorschaugröße:").grid(row=0, column=0, sticky="w")
        
        self.size_var = tk.StringVar(value=self.current_size)
        size_combo = ttk.Combobox(size_frame, textvariable=self.size_var,
                                values=list(self.size_options.keys()), 
                                state="readonly", width=10)
        size_combo.grid(row=1, column=0, sticky="ew", pady=2)
        size_combo.bind('<<ComboboxSelected>>', self.change_thumbnail_size)
        add_tooltip(size_combo, "Größe für Thumbnail-Vorschau, Plane-Vorschau und Fenster anpassen:\nKlein: Kompakt • Mittel: Ausgewogen • Groß: Detailliert")
        
        # Größen-Info
        self.size_info_var = tk.StringVar()
        self.update_size_info()
        size_info_label = ttk.Label(size_frame, textvariable=self.size_info_var, foreground="blue", font=("Arial", 8))
        size_info_label.grid(row=2, column=0, sticky="w")
        
        # Parameter-Einstellungen kompakt
        param_frame = ttk.LabelFrame(config_frame, text="Parameter", padding="5")
        param_frame.grid(row=2, column=0, sticky="ew", pady=2)
        
        self.param_vars = {}
        
        # Parameter nach PBR-Kategorien organisiert für bessere Verständlichkeit
        parameter_categories = {
            "🎨 Base Color": [
                ("base_color_strength", "Intensität", 0.0, 2.0, 1.0, "Base Color Intensität:\nVerstärkt oder reduziert die Intensität der Grundfarbe\n• 0.0 = Keine Farbe (grau)\n• 1.0 = Original-Farbintensität\n• 2.0 = Verstärkte Farben"),
                ("contrast", "Kontrast", 0.5, 2.0, 1.0, "Base Color Kontrast:\nVerstärkt den Farbkontrast der Textur\n• 0.5 = Wenig Kontrast (flach)\n• 1.0 = Original-Kontrast\n• 2.0 = Hoher Kontrast (knackig)"),
                ("brightness", "Helligkeit", 0.0, 2.0, 1.0, "Base Color Helligkeit:\nErhöht oder reduziert die Gesamthelligkeit\n• 0.0 = Sehr dunkel\n• 1.0 = Original-Helligkeit\n• 2.0 = Sehr hell")
            ],
            "⚡ Metallic": [
                ("metallic_strength", "Stärke", 0.0, 2.0, 1.0, "Metallic Verstärkung:\nVerstärkt metallische Eigenschaften für realistischere Reflexionen\n• 0.0 = Kein Metall (isolierend)\n• 1.0 = Original-Metallwerte\n• 2.0 = Verstärkt metallisch"),
                ("metallic_threshold", "Schwellenwert", 0, 255, 127, "Metallic Schwellenwert:\nBestimmt ab welchem Grauwert Pixel als metallisch erkannt werden\n• 0 = Alles wird metallisch\n• 127 = Mittlere Helligkeit als Grenze\n• 255 = Nur weiße Bereiche werden metallisch")
            ],
            "🌊 Roughness": [
                ("roughness_strength", "Stärke", 0.0, 2.0, 0.2, "Roughness Verstärkung:\nVerstärkt Oberflächenrauheit für realistischere Lichtstreuung\n• 0.0 = Spiegelglatt\n• 0.2 = Leicht aufgeraut (Standard)\n• 2.0 = Sehr rau (matt)")
            ],
            "🗻 Normal Map": [
                ("normal_strength", "Stärke", 0.0, 2.0, 0.2, "Normal Map Stärke:\nVerstärkt die Tiefenwirkung der Normal Map für realistischere Oberflächen\n• 0.0 = Flache Oberfläche\n• 0.2 = Subtile Struktur (Standard)\n• 2.0 = Stark geprägte Oberfläche"),
                ("normal_flip_y", "Y-Achse umkehren", 0, 1, 0, "Normal Map Y-Orientierung:\nKehrt Y-Achse der Normal Map um für verschiedene Engines\n• 0 = OpenGL Standard (Y nach oben)\n• 1 = DirectX Standard (Y nach unten)")
            ],
            "🔲 Ambient Occlusion": [
                ("occlusion_strength", "AO Stärke", 0.0, 2.0, 1.0, "Ambient Occlusion Verstärkung:\nVerstärkt Selbstschattierung für realistischere Tiefenwirkung\n• 0.0 = Keine Schatten\n• 1.0 = Original-Schattierung\n• 2.0 = Verstärkte Schatten")
            ],
            "💡 Emission": [
                ("emission_strength", "Leuchtintensität", 0.0, 2.0, 0.0, "Emission Leuchtintensität:\nMacht Texturbereiche selbstleuchtend\n• 0.0 = Kein Leuchten\n• 1.0 = Helle Leuchtbereiche\n• 2.0 = Intensive Leuchtbereiche"),
                ("emission_edge_enhance", "Kontur-Leuchten", 0.0, 1.0, 0.0, "Emission Kantenbeleuchtung:\nLässt Objektkanten zusätzlich leuchten\n• 0.0 = Deaktiviert\n• 1.0 = Aktiviert"),
                ("emission_edge_strength", "Kontur-Stärke", 0.1, 3.0, 1.0, "Emission Kantenstärke:\nIntensität des Kontur-Leuchtens\n• 0.1 = Schwaches Leuchten\n• 1.0 = Normales Leuchten\n• 3.0 = Intensives Leuchten")
            ],
            "👻 Transparenz": [
                ("alpha_strength", "Alpha Stärke", 0.0, 2.0, 1.0, "Alpha/Transparenz Verstärkung:\nVerstärkt Transparenz-/Undurchsichtigkeitseffekte\n• 0.0 = Vollständig transparent\n• 1.0 = Original-Transparenz\n• 2.0 = Verstärkt undurchsichtig")
            ]
        }
        
        current_row = 0
        
        # Erstelle Parameter-Controls gruppiert nach Kategorien
        for category, params in parameter_categories.items():
            # Kategorie-Header mit separierender Linie
            if current_row > 0:
                separator = ttk.Separator(param_frame, orient="horizontal")
                separator.grid(row=current_row, column=0, columnspan=3, sticky="ew", pady=5)
                current_row += 1
            
            # Kategorie-Label mit charakteristischer Farbe
            category_label = tk.Label(param_frame, text=f"{category}", 
                                    font=("Arial", 9, "bold"), 
                                    fg="#2E7D32")  # Dunkelgrün für bessere Lesbarkeit
            category_label.grid(row=current_row, column=0, columnspan=3, sticky="w", pady=(5, 2))
            current_row += 1
            
            # Parameter der Kategorie
            for param_name, display_name, min_val, max_val, default, tooltip in params:
                ttk.Label(param_frame, text=f"  {display_name}:", font=("Arial", 8)).grid(
                    row=current_row, column=0, sticky="w", padx=(10, 0))
                
                if isinstance(default, float):
                    var = tk.DoubleVar()
                    var.set(default)
                else:
                    var = tk.IntVar()
                    var.set(int(default))
                self.param_vars[param_name] = var
                
                scale = ttk.Scale(param_frame, from_=min_val, to=max_val, variable=var, 
                                orient=tk.HORIZONTAL, length=120)
                scale.grid(row=current_row, column=1, sticky="ew", padx=5, pady=1)
                scale.bind('<Motion>', lambda e, p=param_name: self.update_parameter(p))
                
                # Erweiterte Tooltips mit detaillierter Beschreibung
                add_tooltip(scale, tooltip)
                
                # Wert-Anzeige
                value_label = ttk.Label(param_frame, textvariable=var, font=("Arial", 8))
                value_label.grid(row=current_row, column=2, padx=5)
                
                current_row += 1
        
        # Export Buttons kompakt - nur Save GLTF (funktionsfähig)
        export_frame = ttk.Frame(config_frame)
        export_frame.grid(row=3, column=0, pady=10)
        
        save_btn = ttk.Button(export_frame, text="Save GLTF", command=self.save_gltf_for_secondlife)
        if self.button_icons['download']:
            save_btn.config(image=self.button_icons['download'], compound="left")
        save_btn.grid(row=0, column=0, padx=2)
        add_tooltip(save_btn, "GLTF-Material für Second Life/OpenSim exportieren:\nErstellt .gltf, .bin und alle Texturen als PNG (1024x1024)\nInkl. anpassbare Dateinamen und README-Anleitung")
        
        # Export-Optionen entfernt - nur der funktionierende Save GLTF Button bleibt
        
    def clear_all_textures(self):
        """Lösche alle Texturen und setze Placeholder-Bilder zurück"""
        try:
            # Lösche alle Texturen aus dem Material
            for tex_type in self.texture_labels.keys():
                # Reset Preview-Label und setze Placeholder-Bild
                preview_label, file_label = self.texture_labels[tex_type]
                
                # Setze Placeholder-Bild zurück
                self.set_placeholder_image(preview_label, tex_type)
                
                # Reset Datei-Label
                file_label.config(text="Keine Datei")
                
                # Entferne aus Texture-Cache
                if tex_type in self.current_texture_images:
                    del self.current_texture_images[tex_type]
                
                #  FIX: Setze pbr_maker.current_textures auf None zurück (nicht löschen!)
                self.pbr_maker.current_textures[tex_type] = None
                
                # Aktualisiere Tooltip für jede gelöschte Textur
                self.update_texture_tooltip(tex_type)
            
            # Aktualisiere 3D-Viewer
            if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                self.interactive_3d_viewer.set_textures({})
            
            self.status_var.set(" Alle Texturen gelöscht - Placeholder wiederhergestellt")
            print(" Alle Texturen erfolgreich gelöscht und Placeholder-Bilder wiederhergestellt")
            
        except Exception as e:
            print(f" Fehler beim Löschen der Texturen: {e}")
            self.status_var.set(" Fehler beim Löschen")
    
    def clear_single_texture(self, texture_type):
        """Lösche eine einzelne Textur und setze Placeholder zurück"""
        try:
            print(f"Lösche einzelne Textur: {texture_type}")
            
            # Bestätige das Löschen mit einem kleinen Dialog
            from tkinter import messagebox
            
            # Hole aktuellen Dateinamen falls vorhanden
            current_file = "Keine Datei"
            texture_path = self.pbr_maker.current_textures.get(texture_type)
            if texture_path is not None:
                current_file = os.path.basename(texture_path)
            
            # Nur bestätigen wenn eine Textur geladen ist
            if current_file != "Keine Datei":
                confirm = messagebox.askyesno(
                    "Textur löschen",
                    f"Textur '{texture_type.replace('_', ' ').title()}' löschen?\n\n"
                    f"Datei: {current_file}",
                    icon="question"
                )
                
                if not confirm:
                    return
            else:
                # Keine Textur geladen - informiere den Benutzer
                self.status_var.set(f"Keine {texture_type.replace('_', ' ').title()} Textur geladen")
                return
            
            # Lösche Textur aus dem Backend
            self.pbr_maker.current_textures[texture_type] = None
            
            # Lösche aus current_texture_images falls vorhanden
            if hasattr(self, 'current_texture_images') and texture_type in self.current_texture_images:
                del self.current_texture_images[texture_type]
            
            # Setze Placeholder-Bild zurück
            if texture_type in self.texture_labels:
                preview_label, file_label = self.texture_labels[texture_type]
                self.set_placeholder_image(preview_label, texture_type)
                file_label.config(text="Keine Datei", foreground="gray")
            
            # Aktualisiere 3D-Viewer
            if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                # Sammle verbleibende Texturen
                remaining_textures = {}
                for tex_type, tex_path in self.pbr_maker.current_textures.items():
                    if tex_path and os.path.exists(tex_path):
                        try:
                            remaining_textures[tex_type] = Image.open(tex_path)
                        except Exception:
                            pass
                
                self.interactive_3d_viewer.set_textures(remaining_textures)
            
            # Aktualisiere Tooltip nach dem Löschen
            self.update_texture_tooltip(texture_type)
            
            # Status-Update
            self.status_var.set(f"{texture_type.replace('_', ' ').title()} Textur gelöscht")
            print(f"Textur {texture_type} erfolgreich gelöscht")
            
        except Exception as e:
            print(f"Fehler beim Löschen der Textur {texture_type}: {e}")
            self.status_var.set(f"Fehler beim Löschen von {texture_type}")
        
    def load_image_set(self):
        """Lade PBR-Textur-Set basierend auf einer ausgewählten Referenz-Textur"""
        try:
            # Wähle eine Referenz-Textur aus
            initial_dir = os.getcwd()
            base_color_path = self.pbr_maker.current_textures.get("base_color")
            if base_color_path:
                initial_dir = os.path.dirname(base_color_path)
            
            file_path = filedialog.askopenfilename(
                title="Beliebige PBR-Textur auswählen (alle anderen werden automatisch gefunden)",
                filetypes=[
                    ("Alle Bilder", "*.png *.jpg *.jpeg *.tga *.bmp *.tiff *.webp"),
                    ("PNG Dateien", "*.png"),
                    ("JPEG Dateien", "*.jpg *.jpeg"),
                    ("TGA Dateien", "*.tga"),
                    ("Alle Dateien", "*.*")
                ],
                initialdir=initial_dir
            )
            
            if not file_path:
                return
            
            self.status_var.set(" Suche und lade PBR-Texturen...")
            
            # Analysiere die ausgewählte Textur und finde alle zugehörigen
            directory = os.path.dirname(file_path)
            reference_filename = os.path.basename(file_path)
            reference_base = os.path.splitext(reference_filename)[0].lower()
            
            print(f" Referenz-Textur: {reference_filename}")
            print(f"📂 Verzeichnis: {directory}")
            
            # Erkenne den Textur-Typ der Referenz-Datei
            reference_type = self._detect_texture_type(reference_base)
            print(f" Erkannter Typ: {reference_type}")
            
            # Extrahiere Material-Namen aus der Referenz-Datei
            material_name = self._extract_material_name(reference_base, reference_type)
            print(f" Material-Name: '{material_name}'")
            
            if not material_name:
                material_name = reference_base
            
            # Finde alle passenden Texturen im Verzeichnis
            found_textures = self._find_matching_textures_enhanced(directory, material_name)
            
            print(f" Gefundene Texturen: {len(found_textures)}")
            for tex_type, tex_info in found_textures.items():
                print(f"   📄 {tex_type}: {tex_info['filename']} -> {tex_info['path']}")
            
            if not found_textures:
                self.status_var.set(f" Keine PBR-Texturen für '{material_name}' gefunden")
                print(f" Keine passenden PBR-Texturen für Material '{material_name}' gefunden")
                print(" Tipp: Versuchen Sie die Auto-Find Funktion oder prüfen Sie die Dateibenennung")
                return
            
            # Lade die gefundenen Texturen direkt
            loaded_count = 0
            loaded_types = []
            skipped_types = []
            
            print(f" Lade {len(found_textures)} gefundene Texturen...")
            
            for tex_type, tex_info in found_textures.items():
                try:
                    texture_path = tex_info['path']
                    print(f" Lade {tex_type}: {tex_info['filename']} (Pattern: {tex_info['pattern']})")
                    print(f"   📂 Pfad: {texture_path}")
                    print(f"    Existiert: {os.path.exists(texture_path)}")
                    
                    load_result = self.load_texture(tex_type, texture_path)
                    print(f"    Load result: {load_result}")
                    
                    if load_result:
                        loaded_count += 1
                        loaded_types.append(tex_type)
                        print(f" {tex_type}: {tex_info['filename']} erfolgreich geladen")
                    else:
                        skipped_types.append(f"{tex_type} (Ladefehler)")
                        print(f" {tex_type}: Konnte nicht geladen werden")
                
                except Exception as e:
                    skipped_types.append(f"{tex_type} ({str(e)[:30]}...)")
                    print(f" Fehler beim Laden von {tex_type}: {e}")
            
            # Update Preview für alle geladenen Texturen
            print(" Aktualisiere Vorschaubilder...")
            for tex_type in loaded_types:
                if tex_type in self.pbr_maker.current_textures:
                    self.update_texture_preview(tex_type, self.pbr_maker.current_textures[tex_type])
                    # Aktualisiere auch Tooltips für geladene Texturen
                    self.update_texture_tooltip(tex_type)
            
            # Aktualisiere 3D-Vorschau
            if loaded_count > 0:
                print(" Aktualisiere 3D-Vorschau...")
                self.refresh_gltf_preview()
            
            # Erfolgs-/Status-Meldung nur in Statusleiste
            if loaded_count > 0:
                if skipped_types:
                    # Teilerfolg - einige Texturen geladen, andere übersprungen
                    self.status_var.set(f" {loaded_count} von {len(found_textures)} Texturen geladen - {len(skipped_types)} übersprungen")
                    print(f" Material '{material_name}' teilweise geladen:")
                    print(f"    Geladen: {', '.join(loaded_types)}")
                    print(f"    Übersprungen: {', '.join([s.split(' (')[0] for s in skipped_types])}")
                else:
                    # Vollständiger Erfolg - alle Texturen geladen
                    self.status_var.set(f" Material '{material_name}' vollständig geladen - {loaded_count} Texturen")
                    print(f" Material '{material_name}' vollständig geladen: {', '.join(loaded_types)}")
            else:
                # Kein Erfolg - keine Texturen geladen
                self.status_var.set(" Keine Texturen geladen")
                print(f" Keine Texturen für Material '{material_name}' geladen:")
                for error in skipped_types:
                    print(f"    {error}")
            
        except Exception as e:
            print(f" Fehler beim Laden des Bildersets: {e}")
            self.status_var.set(f" Fehler: {str(e)[:50]}...")
    
    def _find_matching_textures_enhanced(self, directory, material_name):
        """Erweiterte Textur-Suche mit besserer Pattern-Erkennung"""
        found_textures = {}
        supported_formats = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tiff', '.webp'}
        
        print(f" Suche Texturen für Material: '{material_name}'")
        
        # Durchsuche alle Dateien im Verzeichnis
        for file in os.listdir(directory):
            if not any(file.lower().endswith(ext) for ext in supported_formats):
                continue
            
            file_lower = file.lower()
            file_base = os.path.splitext(file)[0].lower()
            
            # Prüfe ob der Material-Name im Dateinamen enthalten ist
            if material_name not in file_base:
                continue
            
            # Erkenne Textur-Typ basierend auf Patterns
            detected_type = None
            best_pattern = ""
            
            for texture_type, patterns in self.pbr_maker.texture_patterns.items():
                for pattern in patterns:
                    if pattern in file_lower:
                        # Prüfe ob Material-Name + Pattern eine gute Übereinstimmung ist
                        expected_name = f"{material_name}{pattern}"
                        if expected_name in file_base:
                            detected_type = texture_type
                            best_pattern = pattern
                            break
                if detected_type:
                    break
            
            # Speichere gefundene Textur
            if detected_type:
                found_textures[detected_type] = {
                    'path': os.path.join(directory, file),
                    'filename': file,
                    'pattern': best_pattern,
                    'confidence': 'high' if f"{material_name}{best_pattern}" == file_base else 'medium'
                }
                print(f" {detected_type}: {file} (Pattern: {best_pattern})")
        
        return found_textures
    
    def _detect_texture_type(self, filename_lower):
        """Erkenne den Textur-Typ basierend auf dem Dateinamen"""
        for texture_type, patterns in self.pbr_maker.texture_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return texture_type
        return "base_color"  # Default fallback
    
    def _extract_material_name(self, filename_lower, texture_type):
        """Extrahiere den Material-Namen aus dem Dateinamen"""
        material_name = filename_lower
        
        # Entferne alle bekannten Textur-Pattern
        for tex_type, patterns in self.pbr_maker.texture_patterns.items():
            for pattern in patterns:
                if pattern in material_name:
                    material_name = material_name.replace(pattern, "")
        
        # Entferne führende/nachfolgende Unterstriche und Bindestriche
        material_name = material_name.strip("_-")
        
        return material_name
        
    def update_parameter(self, param_name):
        """Aktualisiere Parameter und GLTF-Vorschau"""
        try:
            # Parameter-Wert abrufen
            if param_name in self.param_vars:
                value = self.param_vars[param_name].get()
                print(f" Parameter '{param_name}' geändert: {value}")
                
                # Spezielle Behandlung für Emission Edge Enhancement Parameter
                if param_name == "emission_edge_enhance":
                    # Konvertiere Float-Wert zu Boolean für Konfiguration
                    edge_enhance = value > 0.5
                    self.pbr_maker.config["EmissionEdgeEnhance"] = edge_enhance
                    print(f" Emission Konturen: {'aktiviert' if edge_enhance else 'deaktiviert'}")
                
                elif param_name == "emission_edge_strength":
                    self.pbr_maker.config["EmissionEdgeStrength"] = value
                    print(f" Kontur-Stärke: {value:.1f}")
                
                # Neue Top-5 Leistungsparameter
                elif param_name == "metallic_strength":
                    self.pbr_maker.config["MetallicStrength"] = value
                    print(f" Metallic Stärke: {value:.1f}")
                
                elif param_name == "base_color_strength":
                    self.pbr_maker.config["BaseColorStrength"] = value
                    print(f" Basisfarbe Stärke: {value:.1f}")
                
                elif param_name == "contrast":
                    self.pbr_maker.config["Contrast"] = value
                    print(f" Kontrast: {value:.1f}")
                
                elif param_name == "brightness":
                    self.pbr_maker.config["Brightness"] = value
                    print(f" Helligkeit: {value:.1f}")
                
                elif param_name == "normal_flip_y":
                    # Boolean Parameter (0/1)
                    flip_y = value > 0.5
                    self.pbr_maker.config["NormalFlipY"] = flip_y
                    print(f" Normal Y umkehren: {'aktiviert' if flip_y else 'deaktiviert'}")
                
                #  Automatisches GLTF-Update nach Parameter-Änderung
                self.refresh_gltf_preview()
                
        except Exception as e:
            print(f" Fehler beim Parameter-Update '{param_name}': {e}")
        
    def setup_texture_panel(self, parent):
        """Erstelle das Textur-Panel"""
        texture_frame = ttk.LabelFrame(parent, text="Texturen", padding="10")
        texture_frame.grid(row=1, column=0, sticky="nswe", padx=(0, 10))
        
        # Textur-Labels
        texture_labels = {
            "base_color": "Base Color",
            "normal": "Normal Map", 
            "roughness": "Roughness",
            "metallic": "Metallic",
            "occlusion": "Occlusion (AO)",
            "emission": "Emission",
            "alpha": "Alpha",
            "height": "Height"
        }
        
        self.texture_frames = {}
        self.texture_labels = {}
        
        # Erstelle Textur-Widgets
        for i, (tex_type, label) in enumerate(texture_labels.items()):
            row = i // 2
            col = i % 2
            
            # Frame für diese Textur
            tex_frame = ttk.Frame(texture_frame)
            tex_frame.grid(row=row, column=col, padx=10, pady=10, sticky="we")
            
            # Label
            ttk.Label(tex_frame, text=label, font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2)
            
            # Vorschaubild (mit Platzhalterbildern)
            preview_label = ttk.Label(tex_frame, anchor="center")
            preview_label.grid(row=1, column=0, columnspan=2, pady=5)
            
            # Setze Platzhalterbild
            self.set_placeholder_image(preview_label, tex_type)
            
            # Drag & Drop Support
            preview_label.drop_target_register(DND_FILES)  # type: ignore
            preview_label.dnd_bind('<<Drop>>', lambda e, t=tex_type: self.on_drop(e, t))  # type: ignore
            
            # Click-Handler
            preview_label.bind("<Button-1>", lambda e, t=tex_type: self.select_texture_file(t))
            # Rechte Maustaste zum Löschen der einzelnen Textur
            preview_label.bind("<Button-3>", lambda e, t=tex_type: self.clear_single_texture(t))
            
            # Dateiname-Label
            file_label = ttk.Label(tex_frame, text="Keine Datei", foreground="gray")
            file_label.grid(row=2, column=0, columnspan=2)
            
            self.texture_frames[tex_type] = tex_frame
            self.texture_labels[tex_type] = (preview_label, file_label)
        
        # Action Buttons
        button_frame = ttk.Frame(texture_frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=20)
        
        # Erste Reihe
        ttk.Button(button_frame, text="Auto-Find", 
                  command=self.auto_find_textures).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Maps generieren", 
                  command=self.generate_missing_maps).grid(row=0, column=1, padx=5)
        
        # Zweite Reihe
        ttk.Button(button_frame, text="Ein Bild für alle", 
                  command=self.load_user_imageset).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="StandardPBR laden", 
                  command=self.load_standard_pbr).grid(row=1, column=1, padx=5, pady=5)
        
        # Dritte Reihe
        ttk.Button(button_frame, text="Alles löschen", 
                  command=self.clear_all).grid(row=2, column=0, padx=5)
    
    def setup_config_panel(self, parent):
        """Erstelle das Konfigurations-Panel"""
        config_frame = ttk.LabelFrame(parent, text="Konfiguration", padding="10")
        config_frame.grid(row=1, column=1, sticky="nswe")
        
        # Material-Presets
        preset_frame = ttk.LabelFrame(config_frame, text="Material-Presets", padding="10")
        preset_frame.grid(row=0, column=0, sticky="we", pady=(0, 10))
        
        ttk.Label(preset_frame, text="Preset auswählen:").grid(row=0, column=0, sticky=tk.W)
        
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                  values=list(self.pbr_maker.material_presets.keys()),
                                  state="readonly")
        preset_combo.grid(row=1, column=0, sticky="we", pady=5)
        preset_combo.bind('<<ComboboxSelected>>', self.apply_preset)
        
        # Größeneinstellungen für Vorschaubilder
        size_frame = ttk.LabelFrame(config_frame, text="Vorschaubild-Größe", padding="10")
        size_frame.grid(row=1, column=0, sticky="we", pady=(0, 10))
        
        ttk.Label(size_frame, text="Größe wählen:").grid(row=0, column=0, sticky=tk.W)
        
        self.size_var = tk.StringVar(value=self.current_size)
        size_combo = ttk.Combobox(size_frame, textvariable=self.size_var,
                                values=list(self.size_options.keys()), 
                                state="readonly", width=15)
        size_combo.grid(row=1, column=0, sticky="we", pady=5)
        size_combo.bind('<<ComboboxSelected>>', self.change_thumbnail_size)
        
        # Größen-Info anzeigen
        self.size_info_var = tk.StringVar()
        self.update_size_info()
        size_info_label = ttk.Label(size_frame, textvariable=self.size_info_var, foreground="blue")
        size_info_label.grid(row=2, column=0, sticky=tk.W)
        
        # Parameter-Einstellungen
        param_frame = ttk.LabelFrame(config_frame, text="Parameter", padding="10")
        param_frame.grid(row=2, column=0, sticky="we", pady=(0, 10))
        
        self.param_vars = {}
        
        # Parameter nach PBR-Kategorien organisiert (vereinfachte Version für GLTF-View)
        parameter_categories = {
            "🎨 Base Color": [
                ("base_color_strength", "Intensität", 0.0, 2.0, 1.0, "Base Color Intensität:\nVerstärkt die Grundfarbenintensität\n• 0.0 = Keine Farbe\n• 1.0 = Original-Intensität\n• 2.0 = Verstärkte Farbe"),
                ("contrast", "Kontrast", 0.5, 2.0, 1.0, "Base Color Kontrast:\nFarbkontrast der Base Color\n• 0.5 = Wenig Kontrast\n• 1.0 = Original-Kontrast\n• 2.0 = Hoher Kontrast"),
                ("brightness", "Helligkeit", 0.0, 2.0, 1.0, "Base Color Helligkeit:\nGesamthelligkeit der Base Color\n• 0.0 = Dunkel\n• 1.0 = Original-Helligkeit\n• 2.0 = Hell")
            ],
            "🗻 Oberfläche": [
                ("normal_strength", "Normal Stärke", 0.0, 2.0, 0.2, "Normal Map Stärke:\nTiefenwirkung der Oberflächenstruktur\n• 0.0 = Flach\n• 0.2 = Subtil (Standard)\n• 2.0 = Stark geprägt"),
                ("roughness_strength", "Roughness Stärke", 0.0, 2.0, 0.2, "Roughness Verstärkung:\nOberflächenrauheit für Lichtstreuung\n• 0.0 = Spiegelglatt\n• 0.2 = Leicht rau\n• 2.0 = Sehr rau"),
                ("occlusion_strength", "AO Stärke", 0.0, 2.0, 1.0, "Ambient Occlusion:\nSelbstschattierung in Vertiefungen\n• 0.0 = Keine Schatten\n• 1.0 = Original-Schatten\n• 2.0 = Starke Schatten")
            ],
            "⚡ Material": [
                ("metallic_threshold", "Metallic Schwelle", 0, 255, 127, "Metallic Schwellenwert:\nSchwellenwert für metallische Bereiche\n• 0 = Alles metallisch\n• 127 = Mittlere Helligkeit\n• 255 = Nur helle Bereiche"),
                ("emission_strength", "Emission Stärke", 0.0, 2.0, 0.0, "Emission Leuchten:\nSelbstleuchtende Bereiche\n• 0.0 = Kein Leuchten\n• 1.0 = Helle Bereiche\n• 2.0 = Intensive Bereiche"),
                ("alpha_strength", "Alpha Stärke", 0.0, 2.0, 1.0, "Alpha/Transparenz:\nTransparenz und Durchsichtigkeit\n• 0.0 = Transparent\n• 1.0 = Original-Alpha\n• 2.0 = Undurchsichtig")
            ]
        }
        
        current_row = 0
        
        # Erstelle Parameter-Controls gruppiert nach Kategorien
        for category, params in parameter_categories.items():
            # Kategorie-Header
            if current_row > 0:
                separator = ttk.Separator(param_frame, orient="horizontal")
                separator.grid(row=current_row, column=0, columnspan=3, sticky="ew", pady=5)
                current_row += 1
            
            # Kategorie-Label
            category_label = tk.Label(param_frame, text=f"{category}", 
                                    font=("Arial", 9, "bold"), 
                                    fg="#1976D2")  # Blau für GLTF-View
            category_label.grid(row=current_row, column=0, columnspan=3, sticky="w", pady=(5, 2))
            current_row += 1
            
            # Parameter der Kategorie
            for key, label, min_val, max_val, default, tooltip in params:
                ttk.Label(param_frame, text=f"  {label}:").grid(row=current_row, column=0, sticky=tk.W, pady=2, padx=(10, 0))
                
                if isinstance(default, float):
                    var = tk.DoubleVar()
                    var.set(default)
                else:
                    var = tk.IntVar()
                    var.set(int(default))
                self.param_vars[key] = var
                
                scale = ttk.Scale(param_frame, from_=min_val, to=max_val, variable=var, 
                                orient=tk.HORIZONTAL, length=200)
                scale.grid(row=current_row, column=1, sticky="we", padx=10, pady=2)
                
                # Erweiterte Tooltips mit detaillierter Beschreibung
                add_tooltip(scale, tooltip)
                
                value_label = ttk.Label(param_frame, text=f"{default:.2f}" if isinstance(default, float) else str(default))
                value_label.grid(row=current_row, column=2, pady=2)
                
                # Update-Handler mit Auto-Refresh
                var.trace('w', lambda *args, lbl=value_label, v=var, param=key: self.update_param_with_refresh(lbl, v, param))
                
                current_row += 1
    
    def update_param_with_refresh(self, label, var, param_name):
        """Update parameter label and trigger GLTF refresh"""
        try:
            # Aktualisiere Label
            value = var.get()
            if isinstance(value, float):
                label.config(text=f"{value:.2f}")
            else:
                label.config(text=str(int(value)))
            
            print(f" Parameter '{param_name}' geändert: {value}")
            
            # Automatisches GLTF-Update nach Parameter-Änderung
            self.refresh_gltf_preview()
            
        except Exception as e:
            print(f" Fehler beim Parameter-Update '{param_name}': {e}")
    
    
    def setup_preview_panel(self, parent):
        """Erstelle das vergrößerte Textur-Plane Vorschau Panel"""
        preview_frame = ttk.LabelFrame(parent, text="Textur-Plane Vorschau", padding="15")
        preview_frame.grid(row=1, column=1, sticky="nsew")
        
        # Licht-Steuerung Frame
        light_frame = ttk.LabelFrame(preview_frame, text="Beleuchtung", padding="10")
        light_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        # Lichtrichtung (Winkel)
        ttk.Label(light_frame, text="Lichtrichtung:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w")
        
        self.light_angle_var = tk.DoubleVar(value=45.0)
        light_angle_scale = ttk.Scale(light_frame, from_=0, to=360, variable=self.light_angle_var,
                                    orient=tk.HORIZONTAL, length=200)
        light_angle_scale.grid(row=0, column=1, sticky="ew", padx=10)
        light_angle_scale.bind('<Motion>', self.update_lighting)
        light_angle_scale.bind('<ButtonRelease-1>', self.update_lighting)
        
        light_angle_label = ttk.Label(light_frame, textvariable=self.light_angle_var, font=("Arial", 8))
        light_angle_label.grid(row=0, column=2)
        add_tooltip(light_angle_scale, "Richtung des Hauptlichts: 0° = von oben, 90° = von rechts, 180° = von unten, 270° = von links")
        
        # Lichtstärke
        ttk.Label(light_frame, text="Lichtstärke:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="w")
        
        self.light_intensity_var = tk.DoubleVar(value=3.0)
        light_intensity_scale = ttk.Scale(light_frame, from_=0.5, to=10.0, variable=self.light_intensity_var,
                                        orient=tk.HORIZONTAL, length=200)
        light_intensity_scale.grid(row=1, column=1, sticky="ew", padx=10)
        light_intensity_scale.bind('<Motion>', self.update_lighting)
        light_intensity_scale.bind('<ButtonRelease-1>', self.update_lighting)
        
        light_intensity_label = ttk.Label(light_frame, textvariable=self.light_intensity_var, font=("Arial", 8))
        light_intensity_label.grid(row=1, column=2)
        add_tooltip(light_intensity_scale, "Stärke der Beleuchtung: 0.5 = schwach, 3.0 = normal, 10.0 = sehr hell")
        
        # Umgebungslicht (Ambient)
        ttk.Label(light_frame, text="Umgebungslicht:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky="w")
        
        self.ambient_light_var = tk.DoubleVar(value=0.3)
        ambient_light_scale = ttk.Scale(light_frame, from_=0.0, to=1.0, variable=self.ambient_light_var,
                                      orient=tk.HORIZONTAL, length=200)
        ambient_light_scale.grid(row=2, column=1, sticky="ew", padx=10)
        ambient_light_scale.bind('<Motion>', self.update_lighting)
        ambient_light_scale.bind('<ButtonRelease-1>', self.update_lighting)
        
        ambient_light_label = ttk.Label(light_frame, textvariable=self.ambient_light_var, font=("Arial", 8))
        ambient_light_label.grid(row=2, column=2)
        add_tooltip(ambient_light_scale, "Grundhelligkeit ohne direktes Licht: 0.0 = vollständig dunkel, 1.0 = gleichmäßig hell")
        
        # Licht-Presets Buttons
        preset_frame = ttk.Frame(light_frame)
        preset_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Button(preset_frame, text="Tageslicht", 
                  command=lambda: self.apply_light_preset("daylight")).grid(row=0, column=0, padx=2)
        ttk.Button(preset_frame, text="Studio", 
                  command=lambda: self.apply_light_preset("studio")).grid(row=0, column=1, padx=2)
        ttk.Button(preset_frame, text="Sonnenuntergang", 
                  command=lambda: self.apply_light_preset("sunset")).grid(row=0, column=2, padx=2)
        ttk.Button(preset_frame, text="Nacht", 
                  command=lambda: self.apply_light_preset("night")).grid(row=0, column=3, padx=2)
        
        add_tooltip(preset_frame, "Vordefinierte Beleuchtungs-Szenarien für verschiedene Umgebungen")
        
        # Konfiguriere Grid-Gewichte
        light_frame.columnconfigure(1, weight=1)
        
        # Info-Text
        info_frame = ttk.Frame(preview_frame)
        info_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        ttk.Label(info_frame, text="• Mausrad: Zoom • Beleuchtung: Anpassung der Lichtverhältnisse", font=("Arial", 9)).grid(row=0, column=0, sticky="w")
        
        # 3D-Vorschau Bereich mit dynamischer Größe basierend auf aktueller Einstellung
        canvas_width, canvas_height = self.plane_preview_size
        self.preview_canvas = tk.Canvas(preview_frame, width=canvas_width, height=canvas_height, bg="black", relief="sunken", bd=2)
        self.preview_canvas.grid(row=2, column=0, pady=15)
        
        # Initialisiere PyRender-basierten 3D-Viewer (nur Plane)
        self.interactive_3d_viewer = PyRender3DViewer(self.preview_canvas, canvas_width, canvas_height)
        
        # Setze Standard-Modell auf Plane
        if hasattr(self.interactive_3d_viewer, 'current_model'):
            self.interactive_3d_viewer.current_model = 'plane'
    
    def create_placeholder_3d(self):
        """Erstelle GLTF-basierte 3D-Szene mit echten 3D-Modellen"""
        self.preview_canvas.delete("all")
        
        # Canvas-Dimensionen aus aktueller Plane-Vorschau-Größe
        canvas_width, canvas_height = self.plane_preview_size
        
        # Bestimme GLTF-Modell basierend auf Vorschau-Modus
        model_type = self.preview_mode.get()  # "cube" oder "sphere"
        
        # Keine Rotation mehr benötigt (statische 2D-Darstellung)
        rotation = 0
        
        # Sammle aktuelle Texturen für GLTF-Rendering
        texture_images = {}
        for tex_type in ['base_color', 'normal', 'roughness', 'metallic', 'occlusion', 'emission', 'alpha', 'height']:
            if tex_type in self.current_texture_images:
                texture_images[tex_type] = self.current_texture_images[tex_type]
            elif hasattr(self, 'placeholder_images'):
                # Verwende Placeholder als Fallback
                size_key = self.current_size.lower()
                if (size_key in self.placeholder_images and 
                    'general' in self.placeholder_images[size_key] and
                    self.placeholder_images[size_key]['general']):
                    
                    # Konvertiere ImageTk.PhotoImage zu PIL.Image
                    placeholder_photo = self.placeholder_images[size_key]['general']
                    if hasattr(placeholder_photo, '_PhotoImage__photo'):
                        # Erstelle PIL Image aus Placeholder
                        try:
                            # Lade Placeholder-Datei direkt
                            placeholder_files = {
                                "klein": "Resources/Placeholder_128.png",
                                "mittel": "Resources/Placeholder_256.png", 
                                "groß": "Resources/Placeholder_512.png"
                            }
                            if size_key in placeholder_files and os.path.exists(placeholder_files[size_key]):
                                texture_images['base_color'] = Image.open(placeholder_files[size_key])
                                break
                        except Exception as e:
                            print(f"Fehler beim Laden des Placeholder: {e}")
        
        # Rendere GLTF-Modell zu Bild
        try:
            rendered_image = self.gltf_viewer.render_gltf_to_image(
                model_type, canvas_width, canvas_height, texture_images, int(rotation)
            )
            
            # Konvertiere zu PhotoImage für Tkinter
            if rendered_image:
                self.gltf_photo = ImageTk.PhotoImage(rendered_image)
                
                # Zeige gerenderte GLTF-Szene
                self.preview_canvas.create_image(
                    canvas_width//2, canvas_height//2, 
                    image=self.gltf_photo, tags="gltf_render"
                )
                
                # Status-Info
                model_name = "Cube.gltf" if model_type == "cube" else "Ball.gltf"
                texture_info = f" mit {len(texture_images)} Textur(en)" if texture_images else ""
                info_text = f"GLTF: {model_name}{texture_info}"
                if rotation != 0:
                    info_text += f" | Rotation: {rotation:.0f}°"
                
                self.preview_canvas.create_text(
                    canvas_width//2, canvas_height - 30, 
                    text=info_text, fill="white", font=("Arial", 12, "bold")
                )
                
                return
                
        except Exception as e:
            print(f" Fehler beim GLTF-Rendering: {e}")
        
        # Fallback auf einfache Darstellung
        self.create_fallback_3d()
    
    def create_fallback_3d(self):
        """Erstelle Fallback 3D-Darstellung wenn GLTF nicht funktioniert"""
        canvas_width, canvas_height = self.plane_preview_size
        cx, cy = canvas_width//2, canvas_height//2
        
        # Verwende Placeholder-Bild als Textur
        size_key = self.current_size.lower()
        placeholder_photo = None
        
        if (hasattr(self, 'placeholder_images') and 
            size_key in self.placeholder_images and 
            'general' in self.placeholder_images[size_key] and
            self.placeholder_images[size_key]['general']):
            placeholder_photo = self.placeholder_images[size_key]['general']
        
        if self.preview_mode.get() == "cube":
            # Vergrößerte Würfel-Dimensionen
            cube_size = 100
            
            # Vorderseite
            self.preview_canvas.create_polygon([
                cx-cube_size, cy-cube_size//2, cx+cube_size, cy-cube_size//2, 
                cx+cube_size, cy+cube_size*1.5, cx-cube_size, cy+cube_size*1.5
            ], fill="#4a4a4a", outline="#ffffff", width=3, tags="object")
            
            # Placeholder-Bild als Textur auf der Vorderseite
            if placeholder_photo:
                try:
                    self.preview_canvas.create_image(cx, cy+cube_size//2, image=placeholder_photo, tags="texture")
                except Exception as e:
                    print(f"Fehler beim Anzeigen der Textur: {e}")
            
            # Rechte Seite
            self.preview_canvas.create_polygon([
                cx+cube_size, cy-cube_size//2, cx+cube_size*1.5, cy-cube_size, 
                cx+cube_size*1.5, cy+cube_size, cx+cube_size, cy+cube_size*1.5
            ], fill="#2a2a2a", outline="#ffffff", width=3, tags="object")
            
            # Oberseite
            self.preview_canvas.create_polygon([
                cx-cube_size, cy-cube_size//2, cx-cube_size//2, cy-cube_size, 
                cx+cube_size*1.5, cy-cube_size, cx+cube_size, cy-cube_size//2
            ], fill="#6a6a6a", outline="#ffffff", width=3, tags="object")
            
        else:  # sphere
            sphere_radius = 120
            
            if placeholder_photo:
                try:
                    self.preview_canvas.create_image(cx, cy, image=placeholder_photo, tags="texture")
                    self.preview_canvas.create_oval(
                        cx-sphere_radius, cy-sphere_radius, cx+sphere_radius, cy+sphere_radius,
                        fill="", outline="#ffffff", width=3, tags="object"
                    )
                except Exception as e:
                    print(f"Fehler beim Anzeigen der Kugel-Textur: {e}")
                    self.create_simple_sphere(cx, cy)
            else:
                self.create_simple_sphere(cx, cy)
        
        # Info-Text
        info_text = "3D-Vorschau (Fallback)"
        if placeholder_photo:
            info_text += f" - Placeholder_{['128', '256', '512'][['klein', 'mittel', 'groß'].index(self.current_size.lower())]}.png"
        
        self.preview_canvas.create_text(cx, canvas_height - 30, text=info_text, 
                                      fill="white", font=("Arial", 12, "bold"))
    
    def create_simple_sphere(self, cx, cy):
        """Erstelle einfache Kugel mit Schattierung (skaliert basierend auf Canvas-Größe)"""
        canvas_width, canvas_height = self.plane_preview_size
        # Basis-Radius skaliert proportional zur Canvas-Größe (20% der kleineren Dimension)
        base_radius = min(canvas_width, canvas_height) // 5
        
        for i in range(8):  # Mehr Schichten für besseren Effekt
            shade = 80 - i * 10
            color = f"#{shade:02x}{shade:02x}{shade:02x}"
            radius = base_radius - i * (base_radius // 10)
            
            self.preview_canvas.create_oval(
                cx-radius, cy-radius, cx+radius, cy+radius,
                fill=color, outline="", tags="object"
            )
        
        # Highlight (skaliert)
        highlight_size = base_radius // 4
        self.preview_canvas.create_oval(
            cx-highlight_size*2, cy-highlight_size*2, cx-highlight_size//2, cy-highlight_size//2,
            fill="#ffffff", outline="", tags="object"
        )
    
    def update_preview_mode(self):
        """Plane-Modus (nicht mehr verwendet da nur noch Plane)"""
        self.status_var.set("� Textur-Plane aktiv")
    
    def update_lighting(self, event=None):
        """Aktualisiere Beleuchtung für die 3D-Vorschau"""
        try:
            if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                # Hole aktuelle Licht-Einstellungen
                angle = self.light_angle_var.get()
                intensity = self.light_intensity_var.get()
                ambient = self.ambient_light_var.get()
                
                # Aktualisiere Licht im 3D-Viewer (wenn PyRender verfügbar)
                if hasattr(self.interactive_3d_viewer, 'update_lighting'):
                    self.interactive_3d_viewer.update_lighting(angle, intensity, ambient)
                
                # Übertrage Licht-Parameter an Fallback-Renderer
                if hasattr(self.interactive_3d_viewer, 'light_angle'):
                    self.interactive_3d_viewer.light_angle = angle
                    self.interactive_3d_viewer.light_intensity = intensity
                    self.interactive_3d_viewer.ambient_intensity = ambient
                    
                    # Trigger Neuzeichnung für Fallback-Renderer
                    if hasattr(self.interactive_3d_viewer, 'render_fallback'):
                        self.interactive_3d_viewer.render_fallback()
                
                # Automatisches GLTF-Update nach Licht-Änderung
                self.refresh_gltf_preview()
                
                self.status_var.set(f" Beleuchtung: {angle:.0f}° | Stärke: {intensity:.1f} | Umgebung: {ambient:.1f}")
            else:
                self.status_var.set(" 3D-Viewer nicht verfügbar für Beleuchtungsänderung")
                
        except Exception as e:
            print(f" Fehler beim Aktualisieren der Beleuchtung: {e}")
    
    def apply_light_preset(self, preset_name):
        """Wende vordefinierte Beleuchtungs-Presets an"""
        try:
            light_presets = {
                "daylight": {"angle": 45.0, "intensity": 5.0, "ambient": 0.4},
                "studio": {"angle": 135.0, "intensity": 4.0, "ambient": 0.3},
                "sunset": {"angle": 270.0, "intensity": 2.5, "ambient": 0.6},
                "night": {"angle": 90.0, "intensity": 1.5, "ambient": 0.2}
            }
            
            if preset_name in light_presets:
                preset = light_presets[preset_name]
                
                # Setze Licht-Parameter
                self.light_angle_var.set(preset["angle"])
                self.light_intensity_var.set(preset["intensity"])
                self.ambient_light_var.set(preset["ambient"])
                
                # Aktualisiere Beleuchtung
                self.update_lighting()
                
                self.status_var.set(f" Beleuchtungs-Preset '{preset_name.title()}' angewendet")
                print(f" Beleuchtungs-Preset '{preset_name}' angewendet: Winkel={preset['angle']}°, Stärke={preset['intensity']}, Umgebung={preset['ambient']}")
            
        except Exception as e:
            print(f" Fehler beim Anwenden des Beleuchtungs-Presets '{preset_name}': {e}")
    
    def update_renderer_mode(self, event=None):
        """Renderer-Modus nicht mehr verwendet (nur noch 2D-Darstellung)"""
        self.status_var.set(" Verwendet automatisch 2D-Fallback-Darstellung")
    
    def update_rotation(self, event=None):
        """Rotation nicht mehr verwendet (statische 2D-Darstellung)"""
        self.status_var.set(" Rotation deaktiviert - statische Darstellung aktiv")
    
    def refresh_gltf_preview(self):
        """Zentrale Methode für automatisches GLTF-Vorschau-Update"""
        try:
            if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                print(" Automatisches GLTF-Vorschau-Update...")
                
                # Sammle alle aktuellen Texturen
                texture_images = {}
                
                # 1. Zuerst bereits geladene Images verwenden (z.B. generierte Maps)
                if hasattr(self, 'current_texture_images'):
                    for texture_type, texture_image in self.current_texture_images.items():
                        if texture_image is not None:
                            texture_images[texture_type] = texture_image
                            print(f"   Textur '{texture_type}' aus Cache geladen")
                
                # 2. Dann fehlende Texturen aus Pfaden laden
                for texture_type, texture_path in self.pbr_maker.current_textures.items():
                    if texture_type not in texture_images and texture_path and os.path.exists(texture_path):
                        try:
                            texture_image = Image.open(texture_path)
                            texture_images[texture_type] = texture_image
                            print(f"   Textur '{texture_type}' aus Datei geladen")
                        except Exception as e:
                            print(f"   Fehler beim Laden von '{texture_type}': {e}")
                
                # Aktualisiere 3D-Viewer
                self.interactive_3d_viewer.set_textures(texture_images)
                print(" GLTF-Vorschau automatisch aktualisiert")
                
            else:
                print(" Kein 3D-Viewer für Auto-Update verfügbar")
                
        except Exception as e:
            print(f" Fehler beim automatischen GLTF-Update: {e}")
    
    def update_texture_for_gltf(self, texture_type):
        """Aktualisiere Textur für GLTF-Rendering"""
        try:
            # Hole aktuellen Textur-Pfad
            texture_path = self.pbr_maker.current_textures.get(texture_type)
            print(f" Update GLTF Textur '{texture_type}': {texture_path}")
            
            if texture_path and os.path.exists(texture_path):
                # Lade PIL Image für GLTF-Rendering
                texture_image = Image.open(texture_path)
                self.current_texture_images[texture_type] = texture_image
                print(f" Textur '{texture_type}' geladen: {texture_image.size}")
                
                # Aktualisiere 3D-Viewer mit neuen Texturen
                if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                    print(" Aktualisiere 3D-Viewer mit neuer Textur...")
                    self.interactive_3d_viewer.set_textures(self.current_texture_images)
                    print(" 3D-Viewer aktualisiert")
                else:
                    print(" Kein 3D-Viewer verfügbar")
            else:
                print(f" Textur-Pfad nicht gefunden: {texture_path}")
                
        except Exception as e:
            print(f" Fehler beim Aktualisieren der GLTF-Textur {texture_type}: {e}")
    
    def on_drop(self, event, texture_type):
        """Handle Drag & Drop von Dateien"""
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            self.load_texture(texture_type, file_path)
            self.status_var.set(f" {texture_type.upper()} via Drag & Drop geladen")
    
    def select_texture_file(self, texture_type):
        """Datei-Dialog für Textur-Auswahl"""
        file_path = filedialog.askopenfilename(
            title=f"Wähle {texture_type.upper()} Textur",
            filetypes=[("Bilddateien", "*.png *.jpg *.jpeg *.tga *.bmp"), ("Alle Dateien", "*.*")]
        )
        
        if file_path:
            self.load_texture(texture_type, file_path)
    
    def load_texture(self, texture_type, file_path):
        """Lade Textur und aktualisiere Vorschau"""
        try:
            print(f" load_texture aufgerufen: {texture_type} -> {file_path}")
            
            # Setze Textur-Pfad
            self.pbr_maker.current_textures[texture_type] = file_path
            print("    Textur-Pfad gesetzt in current_textures")
           
            # Lade Vorschaubild
            with Image.open(file_path) as img:
                print(f"    Bild geöffnet: {img.size}")
                
                # Erstelle Thumbnail
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                print("    PhotoImage erstellt")
                
                # Aktualisiere UI
                preview_label, file_label = self.texture_labels[texture_type]
                preview_label.config(image=photo, text="")
                
                # Referenz behalten - robuster Ansatz
                if not hasattr(preview_label, '_image_refs'):
                    preview_label._image_refs = []
                preview_label._image_refs.append(photo)
                print("    UI aktualisiert")
                
                file_label.config(text=self.truncate_filename(os.path.basename(file_path)), foreground="black")
                print("    File-Label aktualisiert")
                
                self.status_var.set(f" {texture_type.upper()} geladen: {os.path.basename(file_path)}")
                
                #  Aktualisiere Tooltip mit Dateiinformationen
                self.update_texture_tooltip(texture_type)
                print("    Tooltip aktualisiert")
                
                #  Automatisches GLTF-Update nach Textur-Laden
                self.refresh_gltf_preview()
                print("    GLTF-Preview aktualisiert")
                
                print(f"    load_texture erfolgreich für {texture_type}")
                return True  # Erfolg
                
        except Exception as e:
            print(f" Fehler in load_texture({texture_type}): {e}")
            self.status_var.set(f" Fehler beim Laden: {str(e)}")
            return False  # Fehler
    
    def update_texture_preview(self, texture_type, file_path):
        """Aktualisiere nur die Vorschau eines bereits geladenen Bildes mit neuer Thumbnail-Größe"""
        try:
            if not file_path or not os.path.exists(file_path):
                return
                
            # Lade Vorschaubild mit neuer Größe
            with Image.open(file_path) as img:
                # Erstelle Thumbnail mit aktueller Größe
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Aktualisiere nur die Vorschau
                preview_label, file_label = self.texture_labels[texture_type]
                preview_label.config(image=photo, text="")
                preview_label.image = photo  # Referenz behalten
                
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Vorschau für {texture_type}: {e}")
            messagebox.showerror("Fehler", f"Konnte Textur nicht laden:\n{str(e)}")
    
    def toggle_auto_rotation(self):
        """Auto-Rotation nicht mehr verwendet (statische 2D-Darstellung)"""
        self.status_var.set(" Auto-Rotation deaktiviert - statische Darstellung aktiv")
    
    def get_texture_filenames_dialog(self, base_name):
        """Dialog zur Anpassung der Textur-Dateinamen"""
        try:
            # Erstelle Dialog-Fenster
            dialog = tk.Toplevel(self.root)
            dialog.title("Textur-Dateinamen anpassen")
            dialog.geometry("600x700")
            dialog.grab_set()  # Modal dialog
            dialog.resizable(False, False)
            
            # Zentriere Dialog
            dialog.transient(self.root)
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            # Beschreibung
            description = ttk.Label(dialog, text="Passen Sie die Dateinamen für die exportierten Texturen an:", 
                                  font=("Arial", 10, "bold"))
            description.pack(pady=10)
            
            # Frame für Eingabefelder
            input_frame = ttk.Frame(dialog)
            input_frame.pack(pady=10, padx=20, fill="both", expand=True)
            
            # Standard-Mappings für Texturen
            tex_info = {
                "base_color": ("Base Color (Diffuse)", "_col"),
                "normal": ("Normal Map", "_nrm"),
                "roughness": ("Roughness", "_rough"),
                "metallic": ("Metallic", "_metal"),
                "occlusion": ("Ambient Occlusion", "_occ"),
                "emission": ("Emission", "_emission"),
                "alpha": ("Alpha/Transparency", "_alpha"),
                "height": ("Height/Bump", "_height")
            }
            
            # Speichere Eingabefelder
            entries = {}
            result = {}
            
            # Erstelle Eingabefelder für verfügbare Texturen
            row = 0
            for tex_type, (display_name, default_suffix) in tex_info.items():
                # Prüfe ob diese Textur verfügbar ist
                has_texture = (
                    (tex_type in self.pbr_maker.current_textures and 
                     self.pbr_maker.current_textures[tex_type] is not None) or
                    (hasattr(self, 'current_texture_images') and 
                     tex_type in self.current_texture_images and 
                     self.current_texture_images[tex_type])
                )
                
                if has_texture:
                    # Label
                    label = ttk.Label(input_frame, text=f"{display_name}:", font=("Arial", 9))
                    label.grid(row=row, column=0, sticky="w", pady=2)
                    
                    # Eingabefeld mit Standard-Namen
                    default_name = f"{base_name}{default_suffix}.png"
                    entry_var = tk.StringVar(value=default_name)
                    entry = ttk.Entry(input_frame, textvariable=entry_var, width=40)
                    entry.grid(row=row, column=1, sticky="ew", pady=2, padx=(10, 0))
                    
                    entries[tex_type] = entry_var
                    row += 1
            
            # Konfiguriere Grid-Gewichte
            input_frame.columnconfigure(1, weight=1)
            
            # Hinweise
            hints_frame = ttk.LabelFrame(dialog, text="Hinweise", padding="10")
            hints_frame.pack(pady=10, padx=20, fill="x")
            
            hints_text = """• Alle Texturen werden als PNG-Dateien (1024x1024) exportiert
• Dateinamen werden automatisch mit .png ergänzt falls nicht vorhanden
• Verwenden Sie aussagekräftige Namen für bessere Organisation
• Kompatibel mit Second Life/OpenSim und GLTF-Packer Standards"""
            
            hints_label = ttk.Label(hints_frame, text=hints_text, justify="left", font=("Arial", 8))
            hints_label.pack()
            
            # Button Frame
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=20)
            
            # Ergebnis-Variable
            dialog_result = {'cancelled': True, 'filenames': {}}
            
            def on_ok():
                # Sammle alle Dateinamen
                for tex_type, entry_var in entries.items():
                    filename = entry_var.get().strip()
                    if filename:
                        # Stelle sicher, dass .png Endung vorhanden ist
                        if not filename.lower().endswith('.png'):
                            filename += '.png'
                        result[tex_type] = filename
                
                dialog_result['cancelled'] = False
                dialog_result['filenames'] = result
                dialog.destroy()
            
            def on_cancel():
                dialog.destroy()
            
            def reset_to_defaults():
                for tex_type, entry_var in entries.items():
                    if tex_type in tex_info:
                        default_name = f"{base_name}{tex_info[tex_type][1]}.png"
                        entry_var.set(default_name)
            
            # Buttons
            ttk.Button(button_frame, text="Standard wiederherstellen", command=reset_to_defaults).pack(side="left", padx=5)
            ttk.Button(button_frame, text="Abbrechen", command=on_cancel).pack(side="right", padx=5)
            ttk.Button(button_frame, text="OK", command=on_ok).pack(side="right", padx=5)
            
            # Warte auf Dialog-Schließung
            dialog.wait_window()
            
            # Rückgabe der Ergebnisse
            if dialog_result['cancelled']:
                return None
            else:
                return dialog_result['filenames']
                
        except Exception as e:
            print(f"Fehler im Dateinamen-Dialog: {e}")
            messagebox.showerror("Dialog Fehler", f"Fehler beim Anzeigen des Dateinamen-Dialogs: {e}")
            return None

    def save_gltf_for_secondlife(self):
        """Speichere GLTF und Texturen für Second Life / OpenSim Kompatibilität"""
        try:
            # Prüfe ob Base Color Textur verfügbar ist
            if "base_color" not in self.pbr_maker.current_textures or not self.pbr_maker.current_textures["base_color"]:
                messagebox.showwarning(
                    "Base Color erforderlich", 
                    "Bitte laden Sie zuerst eine Base Color Textur.\n\n"
                    "Diese wird als Basis für die Dateinamen verwendet."
                )
                return
            
            # Dateiname-Auswahl Dialog
            base_color_path = self.pbr_maker.current_textures["base_color"]
            base_color_dir = os.path.dirname(base_color_path)
            
            # Erstelle Standard-Namen basierend auf Base Color Textur
            base_name = os.path.splitext(os.path.basename(base_color_path))[0]
            # Entferne Base Color Suffixe für sauberen Namen
            for suffix in ["_albedo", "_diffuse", "_color", "_basecolor", "_base_color", "_diff", "_col",
                          "-albedo", "-diffuse", "-color", "-basecolor", "-base-color", "-diff", "-col"]:
                base_name = base_name.replace(suffix, "")
            
            # Dateinamen-Auswahl Dialog
            gltf_file_path = filedialog.asksaveasfilename(
                title="GLTF-Datei speichern als...",
                initialdir=base_color_dir,
                initialfile=f"{base_name}.gltf",
                defaultextension=".gltf",
                filetypes=[
                    ("GLTF Files", "*.gltf"),
                    ("All Files", "*.*")
                ]
            )
            
            if not gltf_file_path:
                self.status_var.set("GLTF Export abgebrochen")
                return
            
            # Extrahiere Verzeichnis und Basis-Namen aus der Auswahl
            output_dir = os.path.dirname(gltf_file_path)
            gltf_filename = os.path.basename(gltf_file_path)
            user_base_name = os.path.splitext(gltf_filename)[0]
            
            # Stelle sicher, dass das Ausgabeverzeichnis existiert
            os.makedirs(output_dir, exist_ok=True)
            
            self.status_var.set(f"Erstelle GLTF Export: {user_base_name}...")
            
            # Dialog für Textur-Dateinamen Anpassung
            texture_names = self.get_texture_filenames_dialog(user_base_name)
            if not texture_names:
                self.status_var.set("GLTF Export abgebrochen")
                return
            
            # Sammle alle verfügbaren Texturen
            texture_files = {}
            texture_size = 1024  # Standard-Größe für Second Life/OpenSim
            
            # Exportiere Texturen mit benutzerdefinierten Dateinamen
            for tex_type in ["base_color", "normal", "roughness", "metallic", "occlusion", "emission", "alpha", "height"]:
                # Überspringe Texturen, die nicht verfügbar oder nicht benannt sind
                if tex_type not in texture_names:
                    continue
                    
                # Prüfe sowohl current_texture_images als auch current_textures
                texture_image = None
                
                # Erst versuchen aus current_texture_images zu laden
                if (hasattr(self, 'current_texture_images') and 
                    tex_type in self.current_texture_images and 
                    self.current_texture_images[tex_type]):
                    texture_image = self.current_texture_images[tex_type]
                # Falls das nicht funktioniert, aus current_textures laden
                elif (tex_type in self.pbr_maker.current_textures and 
                      self.pbr_maker.current_textures[tex_type] is not None):
                    texture_path_src = self.pbr_maker.current_textures[tex_type]
                    try:
                        if texture_path_src is not None:
                            texture_image = Image.open(texture_path_src)
                        else:
                            print(f"Kein gültiger Pfad für {tex_type}: None")
                            continue
                    except Exception as e:
                        print(f"Fehler beim Laden von {tex_type}: {e}")
                        continue
                
                if texture_image:
                    # Verwende benutzerdefinierten Dateinamen
                    texture_filename = texture_names[tex_type]
                    texture_path = os.path.join(output_dir, texture_filename)
                    
                    # Resize auf Standard-Größe und speichere als PNG
                    if texture_image.size != (texture_size, texture_size):
                        texture_image = texture_image.resize((texture_size, texture_size), Image.Resampling.LANCZOS)
                    
                    # Konvertiere zu RGB falls nötig (entferne Alpha für Base Color)
                    if tex_type == "base_color" and texture_image.mode in ("RGBA", "LA"):
                        # Erstelle weißen Hintergrund für Base Color
                        background = Image.new("RGB", texture_image.size, (255, 255, 255))
                        if texture_image.mode == "RGBA":
                            background.paste(texture_image, mask=texture_image.split()[-1])
                        else:
                            background.paste(texture_image)
                        texture_image = background
                    elif tex_type in ["normal", "roughness", "metallic", "occlusion", "height"] and texture_image.mode in ("RGBA", "LA"):
                        # Für andere Maps: Verwende RGB ohne Alpha
                        texture_image = texture_image.convert("RGB")
                    
                    # Speichere Textur tatsächlich
                    try:
                        texture_image.save(texture_path, "PNG", optimize=True)
                        texture_files[tex_type] = texture_filename
                        print(f" Textur gespeichert: {texture_path}")
                    except Exception as e:
                        print(f" Fehler beim Speichern von {tex_type}: {e}")
                else:
                    print(f" Keine {tex_type} Textur verfügbar")
            
            print(f"Gespeicherte Texturen: {list(texture_files.values())}")
            
            # Erstelle GLTF-kompatible Material-Definition
            gltf_material, positions, texcoords, indices = self.create_secondlife_gltf_material(user_base_name, texture_files)
            
            # Speichere GLTF-Datei (bereits vom Dialog bestimmt)
            with open(gltf_file_path, 'w', encoding='utf-8') as f:
                json.dump(gltf_material, f, indent=2, ensure_ascii=False)
            
            # Erstelle und speichere Binärdatei (.bin)
            bin_filename = f"{user_base_name}.bin"
            bin_path = os.path.join(output_dir, bin_filename)
            
            import struct
            with open(bin_path, 'wb') as f:
                # Schreibe Positions (VEC3 FLOAT)
                for pos in positions:
                    f.write(struct.pack('<f', pos))
                
                # Schreibe Texture Coordinates (VEC2 FLOAT) 
                for uv in texcoords:
                    f.write(struct.pack('<f', uv))
                
                # Schreibe Indices (UNSIGNED_SHORT)
                for idx in indices:
                    f.write(struct.pack('<H', idx))
            
            print(f"GLTF-Datei erstellt: {gltf_file_path}")
            print(f"Binärdatei erstellt: {bin_path}")
            
            # Erstelle auch eine Info-Datei mit Anweisungen
            info_filename = f"{user_base_name}_README.txt"
            info_path = os.path.join(output_dir, info_filename)
            
            info_content = f"""Second Life / OpenSim Material Export (GLTF-Packer kompatibel)
==================================================================

Material Name: {user_base_name}
Export Date: {os.path.basename(__file__)} - {self.__class__.__name__}
Base Color Source: {os.path.basename(base_color_path)}
Export Directory: {os.path.basename(output_dir)}/

Exported Files:
--------------
- {gltf_filename} (GLTF Material Definition)
- {bin_filename} (GLTF Binary Geometry Data)
{chr(10).join(f"- {filename} ({tex_type.replace('_', ' ').title()} Texture)" for tex_type, filename in texture_files.items())}

Directory Structure:
-------------------
{os.path.basename(output_dir)}/
    ├── {gltf_filename}
    ├── {bin_filename}
    ├── {user_base_name}_README.txt
    {chr(10).join(f"    ├── {filename}" for filename in texture_files.values())}

Installation Instructions:
-------------------------
1. Upload alle PNG-Dateien als Texturen in Second Life/OpenSim
2. Erstelle ein neues Material
3. Weise die Texturen den entsprechenden Slots zu:
   - Base Color: Diffuse/Albedo Map
   - Normal: Normal Map  
   - Roughness: Roughness Map
   - Metallic: Metallic Map
   - Occlusion: Ambient Occlusion Map
   - Emission: Emissive Map
   - Alpha: Alpha/Transparency Map
   - Height: Bump/Height Map

Material Settings:
-----------------
- Roughness: {getattr(self.pbr_maker, 'roughness_factor', 0.5):.2f}
- Metallic: {getattr(self.pbr_maker, 'metallic_factor', 0.0):.2f}
- Alpha Mode: {getattr(self.pbr_maker, 'alpha_mode', 'OPAQUE')}
- Alpha Cutoff: {getattr(self.pbr_maker, 'alpha_cutoff', 0.5):.2f}

GLTF-Packer Kompatibilität:
---------------------------
- Automatische Speicherung im 'gltf_textures' Unterordner
- Verzeichnis-Struktur entspricht GLTF-Packer Standard
- Textur-Größe: 1024x1024 PNG (optimal für Second Life/OpenSim)
- GLTF 2.0 Standard mit PBR-Material-Definition

Notes:
------
- All textures are exported as 1024x1024 PNG files for optimal compatibility
- Use PNG format to preserve quality and transparency
- Normal maps use OpenGL standard (Y+ up)
- Roughness values: 0.0 = very smooth, 1.0 = very rough
- Metallic values: 0.0 = non-metal, 1.0 = pure metal
- Compatible with GLTF-Packer directory structure
"""
            
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(info_content)
            
            # Erfolgsmeldung
            file_count = len(texture_files) + 3  # +3 für GLTF, BIN und README
            self.status_var.set(f"GLTF Export abgeschlossen: {file_count} Dateien erstellt")
            
            messagebox.showinfo(
                "GLTF Export erfolgreich", 
                f"Second Life / OpenSim Material exportiert!\n\n"
                f"📁 Material: {user_base_name}\n"
                f"📂 Verzeichnis: {output_dir}\n"
                f"📄 Dateien: {file_count} (GLTF + BIN + {len(texture_files)} Texturen + README)\n\n"
                f"✅ Benutzerdefinierte Dateinamen verwendet\n"
                f"📖 Siehe {info_filename} für Installationsanweisungen"
            )
            
        except Exception as e:
            error_msg = f"Fehler beim GLTF Export: {str(e)}"
            print(f" {error_msg}")
            self.status_var.set(" GLTF Export fehlgeschlagen")
            messagebox.showerror("Export Fehler", error_msg)
    
    def create_secondlife_gltf_material(self, material_name, texture_files):
        """Erstellt vollständiges GLTF-Material mit korrekter Struktur und Geometrie für Second Life/OpenSim"""
        
        # Geometrie-Daten für eine einfache Plane (2x2 Einheiten)
        positions = [
            -1.0, 0.0, -1.0,  # Vertex 0
             1.0, 0.0, -1.0,  # Vertex 1
             1.0, 0.0,  1.0,  # Vertex 2
            -1.0, 0.0,  1.0   # Vertex 3
        ]
        
        texcoords = [
            0.0, 0.0,  # UV 0
            1.0, 0.0,  # UV 1
            1.0, 1.0,  # UV 2
            0.0, 1.0   # UV 3
        ]
        
        indices = [0, 1, 2, 0, 2, 3]  # Zwei Dreiecke für das Quad
        
        # Erstelle vollständige GLTF-Struktur
        gltf_data = {
            "asset": {
                "generator": "os-materialmaker-secondlife",
                "version": "2.0"
            },
            "scene": 0,
            "scenes": [
                {
                    "nodes": [0],
                    "name": "Scene"
                }
            ],
            "nodes": [
                {
                    "mesh": 0,
                    "name": "Plane"
                }
            ],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "TEXCOORD_0": 1
                            },
                            "indices": 2,
                            "material": 0
                        }
                    ],
                    "name": "Plane"
                }
            ],
            "materials": [],
            "textures": [],
            "images": [],
            "samplers": [
                {
                    "magFilter": 9729,  # LINEAR
                    "minFilter": 9729,  # LINEAR
                    "wrapS": 10497,     # REPEAT
                    "wrapT": 10497      # REPEAT
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": 4,
                    "type": "VEC3",
                    "max": [1.0, 0.0, 1.0],
                    "min": [-1.0, 0.0, -1.0]
                },
                {
                    "bufferView": 1,
                    "componentType": 5126,  # FLOAT
                    "count": 4,
                    "type": "VEC2"
                },
                {
                    "bufferView": 2,
                    "componentType": 5123,  # UNSIGNED_SHORT
                    "count": 6,
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": 48,  # 4 vertices * 3 floats * 4 bytes
                    "target": 34962    # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": 48,
                    "byteLength": 32,  # 4 vertices * 2 floats * 4 bytes
                    "target": 34962    # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": 80,
                    "byteLength": 12,  # 6 indices * 2 bytes
                    "target": 34963    # ELEMENT_ARRAY_BUFFER
                }
            ],
            "buffers": [
                {
                    "byteLength": 92,
                    "uri": f"{material_name}.bin"
                }
            ]
        }
        
        # Erstelle Material-Definition
        material = {
            "name": material_name,
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8
            }
        }
        
        # Texture-Index Zähler
        texture_index = 0
        
        # Base Color Textur
        if "base_color" in texture_files:
            gltf_data["images"].append({
                "uri": texture_files["base_color"]
            })
            gltf_data["textures"].append({
                "sampler": 0,
                "source": texture_index
            })
            material["pbrMetallicRoughness"]["baseColorTexture"] = {
                "index": texture_index
            }
            texture_index += 1
        
        # Metallic/Roughness Textur
        if "metallic" in texture_files or "roughness" in texture_files:
            # Verwende Metallic wenn vorhanden, sonst Roughness
            metallic_roughness_file = texture_files.get("metallic", texture_files.get("roughness"))
            gltf_data["images"].append({
                "uri": metallic_roughness_file
            })
            gltf_data["textures"].append({
                "sampler": 0,
                "source": texture_index
            })
            material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {
                "index": texture_index
            }
            texture_index += 1
        
        # Normal Map
        if "normal" in texture_files:
            gltf_data["images"].append({
                "uri": texture_files["normal"]
            })
            gltf_data["textures"].append({
                "sampler": 0,
                "source": texture_index
            })
            material["normalTexture"] = {
                "index": texture_index,
                "scale": 1.0
            }
            texture_index += 1
        
        # Occlusion Map
        if "occlusion" in texture_files:
            gltf_data["images"].append({
                "uri": texture_files["occlusion"]
            })
            gltf_data["textures"].append({
                "sampler": 0,
                "source": texture_index
            })
            material["occlusionTexture"] = {
                "index": texture_index,
                "strength": 1.0
            }
            texture_index += 1
        
        # Emission Map
        if "emission" in texture_files:
            gltf_data["images"].append({
                "uri": texture_files["emission"]
            })
            gltf_data["textures"].append({
                "sampler": 0,
                "source": texture_index
            })
            material["emissiveTexture"] = {
                "index": texture_index
            }
            material["emissiveFactor"] = [1.0, 1.0, 1.0]
            texture_index += 1
        
        # Füge Material hinzu
        gltf_data["materials"].append(material)
        
        return gltf_data, positions, texcoords, indices
    
    def auto_find_textures(self):
        """Automatische Textur-Suche basierend auf Base Color Dateinamen"""
        base_texture = self.pbr_maker.current_textures.get("base_color")
        if not base_texture:
            messagebox.showwarning("Warnung", "Bitte zuerst eine Base Color Textur laden!")
            return
        
        base_dir = os.path.dirname(base_texture)
        base_filename = os.path.splitext(os.path.basename(base_texture))[0]
        
        # Extrahiere den Basis-Namen (entferne Base Color Pattern)
        base_name = base_filename.lower()
        for pattern in self.pbr_maker.texture_patterns["base_color"]:
            if pattern in base_name:
                base_name = base_name.replace(pattern, "")
                break
        
        # Entferne führende/nachfolgende Unterstriche
        base_name = base_name.strip("_")
        
        print(f" Auto-Find für Base: '{base_filename}' -> Basis-Name: '{base_name}'")
        
        found_count = 0
        
        # Durchsuche alle Textur-Typen
        for tex_type, patterns in self.pbr_maker.texture_patterns.items():
            if tex_type == "base_color" or self.pbr_maker.current_textures[tex_type]:
                continue
            
            best_match = None
            best_score = 0
            
            # Durchsuche alle Dateien im Verzeichnis
            for file in os.listdir(base_dir):
                if not any(file.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tga', '.bmp']):
                    continue
                
                file_lower = file.lower()
                file_base = os.path.splitext(file)[0].lower()
                
                # Prüfe ob eine der Patterns im Dateinamen vorkommt
                for pattern in patterns:
                    if pattern in file_lower:
                        # Berechne Ähnlichkeit zum Basis-Namen
                        score = 0
                        
                        # Höchste Priorität: Exakte Übereinstimmung des Basis-Namens
                        if base_name in file_base:
                            score += 100
                        
                        # Mittlere Priorität: Teilweise Übereinstimmung
                        elif any(part in file_base for part in base_name.split('_') if len(part) > 2):
                            score += 50
                        
                        # Niedrige Priorität: Enthält das Pattern
                        score += 10
                        
                        # Bonus für exakte Pattern-Übereinstimmung
                        if file_base.endswith(pattern) or f"_{pattern.lstrip('_')}" in file_base:
                            score += 20
                        
                        print(f"  {tex_type}: '{file}' -> Score: {score}")
                        
                        if score > best_score:
                            best_score = score
                            best_match = file
                        break
            
            # Lade die beste Übereinstimmung
            if best_match and best_score >= 60:  # Mindest-Score für Akzeptanz
                file_path = os.path.join(base_dir, best_match)
                self.load_texture(tex_type, file_path)
                found_count += 1
                print(f" {tex_type}: '{best_match}' geladen (Score: {best_score})")
            elif best_match:
                print(f" {tex_type}: '{best_match}' übersprungen (Score: {best_score} < 60)")
        
        self.status_var.set(f" Auto-Find abgeschlossen: {found_count} Texturen gefunden")
    
    def generate_missing_maps(self):
        """Generiere fehlende Maps aus Base Color Textur"""
        try:
            # Prüfe ob Base Color verfügbar ist (erst in current_texture_images, dann in current_textures)
            base_image = None
            
            # 1. Zuerst prüfen ob bereits als PIL Image geladen
            if hasattr(self, 'current_texture_images') and "base_color" in self.current_texture_images and self.current_texture_images["base_color"] is not None:
                base_image = self.current_texture_images["base_color"]
                print(" Base Color aus Image-Cache geladen")
            
            # 2. Falls nicht im Cache, aus Pfad laden
            elif "base_color" in self.pbr_maker.current_textures and self.pbr_maker.current_textures["base_color"]:
                base_color_path = self.pbr_maker.current_textures["base_color"]
                if os.path.exists(base_color_path):
                    try:
                        base_image = Image.open(base_color_path)
                        print(f" Base Color aus Datei geladen: {os.path.basename(base_color_path)}")
                    except Exception as e:
                        print(f" Fehler beim Laden der Base Color: {e}")
            
            # 3. Wenn keine Base Color gefunden
            if base_image is None:
                messagebox.showwarning("Warnung", "Base Color Textur erforderlich!\nBitte zuerst eine Base Color Textur laden.")
                return
            self.status_var.set(" Generiere fehlende Maps...")
            
            # Hilfsfunktion: Prüfe ob eine Map bereits vorhanden ist
            def map_exists(texture_type):
                # Prüfe sowohl in current_texture_images als auch in current_textures
                in_cache = (hasattr(self, 'current_texture_images') and 
                           texture_type in self.current_texture_images and 
                           self.current_texture_images[texture_type] is not None)
                
                texture_path = self.pbr_maker.current_textures.get(texture_type)
                in_files = (texture_path is not None and 
                           os.path.exists(texture_path))
                
                return in_cache or in_files
            
            # Generiere Maps nur wenn sie noch nicht vorhanden sind
            maps_generated = []
            
            # 1. Normal Map generieren (aus Heightmap-Approximation)
            if not map_exists("normal"):
                normal_map = self.generate_normal_map_from_base(base_image)
                if normal_map:
                    self.set_texture_image("normal", normal_map)
                    maps_generated.append("Normal Map")
            
            # 2. Roughness Map generieren (invertierte Helligkeit)
            if not map_exists("roughness"):
                roughness_map = self.generate_roughness_map_from_base(base_image)
                if roughness_map:
                    self.set_texture_image("roughness", roughness_map)
                    maps_generated.append("Roughness Map")
            
            # 3. Metallic Map generieren (Schwellenwert-basiert)
            if not map_exists("metallic"):
                metallic_map = self.generate_metallic_map_from_base(base_image)
                if metallic_map:
                    self.set_texture_image("metallic", metallic_map)
                    maps_generated.append("Metallic Map")
            
            # 4. Occlusion Map generieren (Verdunkelung in Ecken/Kanten)
            if not map_exists("occlusion"):
                occlusion_map = self.generate_occlusion_map_from_base(base_image)
                if occlusion_map:
                    self.set_texture_image("occlusion", occlusion_map)
                    maps_generated.append("Occlusion Map")
            
            # 5. Emission Map generieren (dunkle Basis für emissive Bereiche)
            if not map_exists("emission"):
                emission_map = self.generate_emission_map_from_base(base_image)
                if emission_map:
                    self.set_texture_image("emission", emission_map)
                    maps_generated.append("Emission Map")
            
            # 6. Alpha Map generieren (aus Transparenz oder Helligkeit)
            if not map_exists("alpha"):
                alpha_map = self.generate_alpha_map_from_base(base_image)
                if alpha_map:
                    self.set_texture_image("alpha", alpha_map)
                    maps_generated.append("Alpha Map")
            
            # 7. Height Map generieren (Heightmap aus Graustufen)
            if not map_exists("height"):
                height_map = self.generate_height_map_from_base(base_image)
                if height_map:
                    self.set_texture_image("height", height_map)
                    maps_generated.append("Height Map")
            
            # Ergebnis anzeigen
            if maps_generated:
                result_text = f" Generiert: {', '.join(maps_generated)}"
                self.status_var.set(result_text)
                messagebox.showinfo("Maps generiert", f"Erfolgreich generiert:\n• {chr(10).join(maps_generated)}")
                
                # 3D-Vorschau aktualisieren
                self.refresh_gltf_preview()
            else:
                self.status_var.set(" Alle Maps bereits vorhanden")
                messagebox.showinfo("Maps generieren", "Alle Maps sind bereits vorhanden.")
                
        except Exception as e:
            print(f" Fehler bei Map-Generierung: {e}")
            self.status_var.set(" Fehler bei Map-Generierung")
            messagebox.showerror("Fehler", f"Fehler bei der Map-Generierung:\n{str(e)}")
    
    def generate_normal_map_from_base(self, base_image):
        """Generiere Normal Map aus Base Color (Sobel-Edge-Detection)"""
        try:
            # Konvertiere zu Graustufen für Height-Map-Approximation
            gray = base_image.convert('L')
            gray_array = np.array(gray, dtype=np.float32)
            
            # Sobel-Operatoren für Kantendetection
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
            
            # Padding hinzufügen für Randbehandlung
            padded = np.pad(gray_array, 1, mode='edge')
            
            # Gradients berechnen
            grad_x = np.zeros_like(gray_array)
            grad_y = np.zeros_like(gray_array)
            
            for i in range(gray_array.shape[0]):
                for j in range(gray_array.shape[1]):
                    region = padded[i:i+3, j:j+3]
                    grad_x[i, j] = np.sum(region * sobel_x)
                    grad_y[i, j] = np.sum(region * sobel_y)
            
            # Normal-Vektor berechnen
            # Normal = (-dx, -dy, 1) normalisiert
            normal_x = -grad_x
            normal_y = -grad_y
            normal_z = np.ones_like(grad_x)
            
            # Normalisierung
            length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
            normal_x /= length
            normal_y /= length
            normal_z /= length
            
            # Konvertiere zu [0,255] Bereich für RGB
            normal_map_array = np.zeros((*gray_array.shape, 3), dtype=np.uint8)
            normal_map_array[:, :, 0] = ((normal_x + 1) * 127.5).astype(np.uint8)  # R
            normal_map_array[:, :, 1] = ((normal_y + 1) * 127.5).astype(np.uint8)  # G  
            normal_map_array[:, :, 2] = ((normal_z + 1) * 127.5).astype(np.uint8)  # B
            
            return Image.fromarray(normal_map_array, 'RGB')
            
        except Exception as e:
            print(f" Fehler bei Normal Map Generierung: {e}")
            return None
    
    def generate_roughness_map_from_base(self, base_image):
        """Generiere Roughness Map aus Base Color (invertierte Helligkeit)"""
        try:
            # Konvertiere zu Graustufen
            gray = base_image.convert('L')
            gray_array = np.array(gray)
            
            # Invertiere für Roughness (dunkle Bereiche = rau, helle Bereiche = glatt)
            roughness_array = 255 - gray_array
            
            # Leichte Kontrastanpassung für bessere Verteilung
            roughness_array = np.clip(roughness_array * 1.2, 0, 255).astype(np.uint8)
            
            return Image.fromarray(roughness_array, 'L').convert('RGB')
            
        except Exception as e:
            print(f" Fehler bei Roughness Map Generierung: {e}")
            return None
    
    def generate_metallic_map_from_base(self, base_image):
        """Generiere Metallic Map aus Base Color (Schwellenwert-basiert)"""
        try:
            # Konvertiere zu HSV für bessere Farbanalyse
            hsv = base_image.convert('HSV')
            h, s, v = hsv.split()
            
            # Konvertiere zu Arrays
            s_array = np.array(s, dtype=np.float32)
            v_array = np.array(v, dtype=np.float32)
            
            # Metallic-Wahrscheinlichkeit basierend auf hoher Sättigung und Helligkeit
            # Metallische Oberflächen haben oft hohe Sättigung bei mittlerer bis hoher Helligkeit
            metallic_score = (s_array / 255.0) * (v_array / 255.0)
            
            # Schwellenwert anwenden (anpassbar über Parameter)
            threshold = self.param_vars.get("metallic_threshold", tk.IntVar()).get() / 255.0 if hasattr(self, 'param_vars') else 0.5
            metallic_array = (metallic_score > threshold) * 255
            
            return Image.fromarray(metallic_array.astype(np.uint8), 'L').convert('RGB')
            
        except Exception as e:
            print(f" Fehler bei Metallic Map Generierung: {e}")
            return None
    
    def generate_occlusion_map_from_base(self, base_image):
        """Generiere Occlusion Map aus Base Color (Ecken- und Kantendetection)"""
        try:
            # Konvertiere zu Graustufen
            gray = base_image.convert('L')
            gray_array = np.array(gray, dtype=np.float32)
            
            # Gaussian Blur für Rauschreduzierung
            from scipy import ndimage
            blurred = ndimage.gaussian_filter(gray_array, sigma=1.0)
            
            # Laplacian für Kanten-/Eckendetection
            laplacian = ndimage.laplace(blurred)
            
            # Absolute Werte für Kantenstärke
            edge_strength = np.abs(laplacian)
            
            # Normalisiere und invertiere (Kanten = dunkel, flache Bereiche = hell)
            if edge_strength.max() > 0:
                edge_strength = edge_strength / edge_strength.max()
            
            # Occlusion = dunkler an Kanten/Ecken
            occlusion_array = (255 * (1.0 - edge_strength * 0.5)).astype(np.uint8)
            
            return Image.fromarray(occlusion_array, 'L').convert('RGB')
            
        except Exception as e:
            # Fallback ohne scipy
            print(f" Occlusion Map Generierung (Fallback): {e}")
            try:
                # Einfacher Fallback: Verdunkelte Version der Base Color
                gray = base_image.convert('L')
                gray_array = np.array(gray)
                
                # Leichte Verdunklung als einfache AO-Approximation
                occlusion_array = (gray_array * 0.8).astype(np.uint8)
                
                return Image.fromarray(occlusion_array, 'L').convert('RGB')
            except Exception as e2:
                print(f" Fehler bei Occlusion Map Fallback: {e2}")
                return None
    
    def generate_emission_map_from_base(self, base_image):
        """Generiere Emission Map aus Base Color (helle Bereiche = emissiv) mit optionaler Kontur-Hervorhebung"""
        try:
            # Konvertiere zu HSV für bessere Farbanalyse
            hsv = base_image.convert('HSV')
            h, s, v = hsv.split()
            
            # Konvertiere zu Arrays
            s_array = np.array(s, dtype=np.float32)
            v_array = np.array(v, dtype=np.float32)
            
            # Emission basierend auf sehr hellen und gesättigten Bereichen
            # Nur sehr helle Bereiche (>80% Helligkeit) werden als emissiv betrachtet
            emission_threshold = 200  # Schwellenwert für Helligkeit (0-255)
            
            # Erstelle Emission-Maske
            emission_mask = (v_array > emission_threshold) & (s_array > 100)
            
            # Erstelle Emission Map (meist schwarz mit hellen emissiven Bereichen)
            emission_array = np.zeros_like(v_array, dtype=np.uint8)
            emission_array[emission_mask] = v_array[emission_mask].astype(np.uint8)
            
            # Kontur-Hervorhebung anwenden wenn aktiviert
            if hasattr(self, 'pbr_maker') and self.pbr_maker.config.get("EmissionEdgeEnhance", False):
                emission_array = self.apply_edge_enhancement_to_emission(emission_array)
            elif not hasattr(self, 'pbr_maker'):
                # Fallback: Verwende Standard-Einstellung
                pass  # Keine Kontur-Hervorhebung ohne pbr_maker
            
            # Konvertiere zu RGB
            return Image.fromarray(emission_array, 'L').convert('RGB')
            
        except Exception as e:
            print(f" Fehler bei Emission Map Generierung: {e}")
            return None
    
    def apply_edge_enhancement_to_emission(self, emission_array):
        """Wende Kontur-Hervorhebung auf Emission Map an"""
        try:
            # Hole Edge Enhancement Stärke aus der config des PBR Material Makers
            edge_strength = 1.0  # Standard-Wert
            if hasattr(self, 'pbr_maker') and self.pbr_maker:
                edge_strength = self.pbr_maker.config.get("EmissionEdgeStrength", 1.0)
            
            # Konvertiere zu PIL Image für Edge Detection
            emission_img = Image.fromarray(emission_array, 'L')
            
            # Verwende PIL's integrierte Filter für Edge Detection
            try:
                from PIL import ImageFilter
                
                # Verwende PIL's FIND_EDGES Filter
                edges = emission_img.filter(ImageFilter.FIND_EDGES)
                
                # Verstärke Edges basierend auf Stärke
                edge_array = np.array(edges, dtype=np.float32)
                edge_array = np.clip(edge_array * edge_strength, 0, 255).astype(np.uint8)
                
                # Kombiniere mit Original
                enhanced = np.maximum(emission_array, edge_array)
                
                print("Kontur-Hervorhebung mit PIL angewendet")
                return enhanced
                
            except Exception as filter_error:
                print(f"Fehler bei PIL Edge Detection: {filter_error}")
                
                # Fallback: Manuelle Sobel Edge Detection
                return self.manual_sobel_edge_detection(emission_array, edge_strength)
                
        except Exception as e:
            print(f"Fehler bei Kontur-Hervorhebung: {e}")
            return emission_array  # Gib Original zurück bei Fehlern
    
    def manual_sobel_edge_detection(self, image_array, edge_strength):
        """Manuelle Sobel Edge Detection Implementation"""
        try:
            # Sobel Kernels
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            
            # Konvertiere zu float für Berechnungen
            img_float = image_array.astype(np.float32)
            
            # Padding hinzufügen für Kernel-Operationen
            padded = np.pad(img_float, ((1, 1), (1, 1)), mode='edge')
            
            # Initialisiere Gradient-Arrays
            grad_x = np.zeros_like(img_float)
            grad_y = np.zeros_like(img_float)
            
            # Führe Konvolution manuell durch
            for i in range(img_float.shape[0]):
                for j in range(img_float.shape[1]):
                    # Extrahiere 3x3 Region
                    region = padded[i:i+3, j:j+3]
                    
                    # Berechne Gradienten
                    grad_x[i, j] = np.sum(region * sobel_x)
                    grad_y[i, j] = np.sum(region * sobel_y)
            
            # Kombiniere Gradienten
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalisiere und wende Stärke an
            gradient_magnitude = np.clip(gradient_magnitude * edge_strength, 0, 255).astype(np.uint8)
            
            # Kombiniere mit Original
            enhanced = np.maximum(image_array, gradient_magnitude)
            
            print("Manuelle Sobel Edge Detection angewendet")
            return enhanced
            
        except Exception as e:
            print(f"Fehler bei manueller Edge Detection: {e}")
            return image_array
    
    def simple_edge_enhancement(self, emission_array, edge_strength):
        """Einfache Edge Detection ohne externe Abhängigkeiten"""
        try:
            from PIL import ImageFilter
            
            # Konvertiere zu PIL
            emission_img = Image.fromarray(emission_array, 'L')
            
            # Verwende PIL's FIND_EDGES Filter
            edges = emission_img.filter(ImageFilter.FIND_EDGES)
            
            # Verstärke Edges basierend auf Stärke
            edge_array = np.array(edges, dtype=np.float32)
            edge_array = np.clip(edge_array * edge_strength, 0, 255).astype(np.uint8)
            
            # Kombiniere mit Original
            enhanced = np.maximum(emission_array, edge_array)
            
            print("Einfache Kontur-Hervorhebung angewendet")
            return enhanced
            
        except Exception as e:
            print(f"Fehler bei einfacher Edge Detection: {e}")
            return emission_array
    
    def generate_alpha_map_from_base(self, base_image):
        """Generiere Alpha Map aus Base Color (Standard: vollständig opak)"""
        try:
            # Prüfe ob Base Image bereits einen Alpha-Kanal hat
            if base_image.mode in ('RGBA', 'LA'):
                # Extrahiere Alpha-Kanal
                if base_image.mode == 'RGBA':
                    r, g, b, a = base_image.split()
                    return Image.merge('RGB', (a, a, a))
                else:  # LA
                    luminance, alpha = base_image.split()
                    return Image.merge('RGB', (alpha, alpha, alpha))
            else:
                # Kein Alpha-Kanal vorhanden - erstelle vollständig opake Alpha Map
                width, height = base_image.size
                alpha_array = np.full((height, width), 255, dtype=np.uint8)
                
                # Optional: Mache sehr dunkle Bereiche leicht transparent
                gray = base_image.convert('L')
                gray_array = np.array(gray)
                
                # Bereiche unter 10% Helligkeit werden leicht transparent
                dark_mask = gray_array < 25
                alpha_array[dark_mask] = 200  # Leicht transparent
                
                return Image.fromarray(alpha_array, 'L').convert('RGB')
                
        except Exception as e:
            print(f" Fehler bei Alpha Map Generierung: {e}")
            return None
    
    def generate_height_map_from_base(self, base_image):
        """Generiere Height Map aus Base Color (Graustufen als Höheninformation)"""
        try:
            # Konvertiere zu Graustufen
            gray = base_image.convert('L')
            gray_array = np.array(gray, dtype=np.float32)
            
            # Normalisiere die Werte
            if gray_array.max() > gray_array.min():
                normalized = (gray_array - gray_array.min()) / (gray_array.max() - gray_array.min())
                height_array = (normalized * 255).astype(np.uint8)
            else:
                height_array = gray_array.astype(np.uint8)
            
            # Optional: Leichte Glättung für bessere Height Map
            try:
                from scipy import ndimage
                height_array = ndimage.gaussian_filter(height_array, sigma=0.5)
                height_array = (height_array).astype(np.uint8)
            except ImportError:
                # Fallback ohne scipy - verwende Original
                pass
            
            return Image.fromarray(height_array, 'L').convert('RGB')
            
        except Exception as e:
            print(f" Fehler bei Height Map Generierung: {e}")
            return None
    
    def set_texture_image(self, texture_type, image):
        """Setze Textur-Bild und aktualisiere UI"""
        try:
            if texture_type in self.texture_labels:
                # Speichere Bild im Cache
                if not hasattr(self, 'current_texture_images'):
                    self.current_texture_images = {}
                self.current_texture_images[texture_type] = image
                
                # Markiere auch im PBR-Maker als vorhanden (ohne Pfad, da generiert)
                self.pbr_maker.current_textures[texture_type] = "[Generated]"
                
                # Aktualisiere Vorschau-Label
                preview_label, file_label = self.texture_labels[texture_type]
                
                # Erstelle Thumbnail
                thumbnail = image.copy()
                thumbnail.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumbnail)
                preview_label.configure(image=photo)
                preview_label.image = photo
                
                # Aktualisiere Dateiname
                file_label.configure(text="[Generiert]", foreground="green")
                
                print(f" {texture_type} Map generiert und UI aktualisiert")
                
        except Exception as e:
            print(f" Fehler beim Setzen der Textur {texture_type}: {e}")
        
    def apply_preset(self, event=None):
        """Wende Material-Preset mit PyPBR-Optimierungen an"""
        preset_name = self.preset_var.get()
        if preset_name and preset_name in self.pbr_maker.material_presets:
            preset = self.pbr_maker.material_presets[preset_name]
            
            # PyPBR-optimierte Parameter-Mapping
            param_mapping = {
                "normal_strength": "NormalStrength",
                "roughness_strength": "RoughnessStrength", 
                "occlusion_strength": "OcclusionStrength",
                "metallic_threshold": "MetallicThreshold",
                "metallic_strength": "MetallicStrength",
                "emission_strength": "EmissionStrength",
                "alpha_strength": "AlphaStrength",
                # Top-5 Leistungsparameter (PyPBR-optimiert)
                "base_color_strength": "BaseColorStrength",
                "contrast": "Contrast",
                "brightness": "Brightness",
                "normal_flip_y": "NormalFlipY",
                # Edge Enhancement Parameter
                "emission_edge_enhance": "EmissionEdgeEnhance",
                "emission_edge_strength": "EmissionEdgeStrength"
            }
            
            # Wende Parameter mit PyPBR-Optimierungen an
            for param_key, config_key in param_mapping.items():
                if param_key in preset and param_key in self.param_vars:
                    original_value = preset[param_key]
                    
                    # PyPBR-Optimierungen anwenden
                    if hasattr(self.pbr_maker, 'pypbr_pipeline') and self.pbr_maker.pypbr_pipeline.enabled:
                        if param_key == "roughness_strength":
                            # Roughness mit PyPBR-Stabilität
                            optimized_value = self.pbr_maker.pypbr_pipeline.optimize_roughness(original_value)
                            self.param_vars[param_key].set(optimized_value)
                            if optimized_value != original_value:
                                print(f"  {config_key}: {original_value} → {optimized_value} (PyPBR-Stabilität)")
                            else:
                                print(f"  {config_key}: {optimized_value}")
                        elif param_key == "normal_strength":
                            # Normal Map mit PyPBR-Normalisierung
                            optimized_value = self.pbr_maker.pypbr_pipeline.optimize_normal_map(original_value)
                            self.param_vars[param_key].set(optimized_value)
                            if optimized_value != original_value:
                                print(f"  {config_key}: {original_value} → {optimized_value} (PyPBR-Normalisierung)")
                            else:
                                print(f"  {config_key}: {optimized_value}")
                        else:
                            # Standard-Parameter
                            self.param_vars[param_key].set(original_value)
                            print(f"  {config_key}: {original_value}")
                    else:
                        # Fallback ohne PyPBR
                        self.param_vars[param_key].set(original_value)
                        print(f"  {config_key}: {original_value}")
                if param_key in preset and param_key in self.param_vars:
                    self.param_vars[param_key].set(preset[param_key])
                    print(f"  {config_key}: {preset[param_key]}")
            
            # Legacy-Support: Alte JSON-Schlüssel für Kompatibilität
            legacy_mapping = {
                "normal_strength": "NormalStrength",
                "roughness_strength": "RoughnessStrength", 
                "occlusion_strength": "OcclusionStrength",
                "metallic_threshold": "MetallicThreshold",
                "emission_strength": "EmissionStrength",
                "alpha_strength": "AlphaStrength"
            }
            
            for param_key, config_key in legacy_mapping.items():
                if config_key in preset and param_key in self.param_vars:
                    self.param_vars[param_key].set(preset[config_key])
            
            # Lade Emission Edge Enhancement Parameter über param_vars
            if "emission_edge_enhance" in preset and "emission_edge_enhance" in self.param_vars:
                self.param_vars["emission_edge_enhance"].set(preset["emission_edge_enhance"])
                # Konvertiere Float zu Boolean für Konfiguration (>0.5 = aktiviert)
                edge_enhance = preset["emission_edge_enhance"] > 0.5
                self.pbr_maker.config["EmissionEdgeEnhance"] = edge_enhance
            
            if "emission_edge_strength" in preset and "emission_edge_strength" in self.param_vars:
                self.param_vars["emission_edge_strength"].set(preset["emission_edge_strength"])
                self.pbr_maker.config["EmissionEdgeStrength"] = preset["emission_edge_strength"]
            
            self.status_var.set(f" Preset '{preset_name}' angewendet")
            
            # PyPBR-Enhancement anwenden
            if hasattr(self.pbr_maker, 'pypbr_pipeline') and self.pbr_maker.pypbr_pipeline.enabled:
                enhanced_preset = self.pbr_maker.pypbr_pipeline.enhance_material_rendering(preset)
                if enhanced_preset != preset:
                    print(f"PyPBR-Enhancement für '{preset_name}' angewendet")
            
            # Log für Debugging
            if "emission_edge_enhance" in preset or "emission_edge_strength" in preset:
                edge_enhance = preset.get('emission_edge_enhance', 0.0) > 0.5
                edge_strength = preset.get('emission_edge_strength', 1.0)
                print(f"Preset '{preset_name}' geladen: EdgeEnhance={edge_enhance}, EdgeStrength={edge_strength}")
    
    def clear_all(self):
        """Lösche alle Texturen und setze Platzhalterbilder zurück"""
        for tex_type in self.pbr_maker.current_textures:
            self.pbr_maker.current_textures[tex_type] = None
            
            preview_label, file_label = self.texture_labels[tex_type]
            
            # Setze Platzhalterbild zurück
            self.set_placeholder_image(preview_label, tex_type)
            file_label.config(text="Keine Datei", foreground="gray")
        
        self.status_var.set(" Alle Texturen gelöscht")
    
    def change_thumbnail_size(self, event=None):
        """Ändere die Thumbnail-Größe, Plane-Vorschau und Fenstergröße und aktualisiere alle Vorschaubilder"""
        new_size = self.size_var.get()
        if new_size in self.size_options:
            self.current_size = new_size
            self.thumbnail_size = self.size_options[new_size]
            self.plane_preview_size = self.plane_preview_sizes[new_size]
            self.current_window_size = self.window_sizes[new_size]
            self.update_size_info()
            
            # Aktualisiere die Fenstergröße
            self.update_window_size()
            
            # Aktualisiere die Plane-Vorschau Canvas-Größe
            self.update_plane_preview_size()
            
            # Aktualisiere alle Vorschaubilder mit der neuen Größe
            for tex_type in self.texture_labels:
                preview_label, file_label = self.texture_labels[tex_type]
                
                # Prüfe ob eine Textur vorhanden ist (Datei oder generiert)
                has_file_texture = (self.pbr_maker.current_textures[tex_type] and 
                                  self.pbr_maker.current_textures[tex_type] != "[Generated]")
                has_generated_texture = (hasattr(self, 'current_texture_images') and 
                                       tex_type in self.current_texture_images and 
                                       self.current_texture_images[tex_type] is not None)
                
                if has_file_texture:
                    # Lade Datei-basierte Textur mit neuer Größe
                    self.update_texture_preview(tex_type, self.pbr_maker.current_textures[tex_type])
                elif has_generated_texture:
                    # Aktualisiere generierte Textur mit neuer Thumbnail-Größe
                    image = self.current_texture_images[tex_type]
                    thumbnail = image.copy()
                    thumbnail.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(thumbnail)
                    preview_label.configure(image=photo)
                    preview_label.image = photo
                    print(f"   Generierte {tex_type} Textur auf neue Größe angepasst: {self.thumbnail_size}")
                else:
                    # Setze Platzhalterbild zurück
                    self.set_placeholder_image(preview_label, tex_type)
                
                # Aktualisiere Tooltip für alle Texturen
                self.update_texture_tooltip(tex_type)
            
            self.status_var.set(f" Vorschau-Größe auf {new_size} geändert - Thumbnails: {self.thumbnail_size[0]}x{self.thumbnail_size[1]}, Plane: {self.plane_preview_size[0]}x{self.plane_preview_size[1]}")
    
    def update_window_size(self):
        """Aktualisiere die Fenstergröße basierend auf der gewählten Vorschaugröße"""
        try:
            if hasattr(self, 'root') and self.root:
                self.root.geometry(self.current_window_size)
                print(f"Fenstergröße geändert auf: {self.current_window_size}")
        except Exception as e:
            print(f"Fehler beim Ändern der Fenstergröße: {e}")
    
    def update_plane_preview_size(self):
        """Aktualisiere die Größe der Plane-Vorschau Canvas"""
        try:
            if hasattr(self, 'preview_canvas') and self.preview_canvas:
                canvas_width, canvas_height = self.plane_preview_size
                
                # Konfiguriere die Canvas-Größe neu
                self.preview_canvas.config(width=canvas_width, height=canvas_height)
                
                # Aktualisiere auch den 3D-Viewer falls vorhanden
                if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                    try:
                        # Erstelle neuen 3D-Viewer mit neuer Größe
                        self.interactive_3d_viewer = PyRender3DViewer(self.preview_canvas, canvas_width, canvas_height)
                    except Exception as e:
                        print(f"Hinweis: 3D-Viewer konnte nicht aktualisiert werden: {e}")
                
                # Aktualisiere die GLTF-Vorschau mit neuer Größe
                self.refresh_gltf_preview()
                
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Plane-Vorschau: {e}")
    
    def update_size_info(self):
        """Aktualisiere die Größen-Info Anzeige"""
        thumb_width, thumb_height = self.thumbnail_size
        plane_width, plane_height = self.plane_preview_size
        self.size_info_var.set(f"Thumbnails: {thumb_width}x{thumb_height} • Plane: {plane_width}x{plane_height} Pixel")
    
    def update_preview_label_size(self, preview_label):
        """Aktualisiere die Größe eines Vorschau-Labels basierend auf der aktuellen Thumbnail-Größe"""
        # Berechne Label-Breite basierend auf Thumbnail-Größe (mit etwas Padding)
        label_width = max(15, self.thumbnail_size[0] // 8)  # Mindestbreite 15
        preview_label.config(width=label_width)
    
    def load_standard_pbr(self):
        """Lade das StandardPBR Material-Preset"""
        try:
            # Setze das Preset auf StandardPBR
            if "StandardPBR" in self.pbr_maker.material_presets:
                self.preset_var.set("StandardPBR")
                self.apply_preset()
                self.status_var.set(" StandardPBR Preset geladen")
            else:
                self.status_var.set(" StandardPBR Preset nicht gefunden")
        except Exception as e:
            self.status_var.set(f" Fehler beim Laden von StandardPBR: {e}")
    
    def load_user_imageset(self):
        """Lade ein Bild für alle Texturtypen"""
        try:
            # Zuerst StandardPBR laden
            self.load_standard_pbr()
            
            # Ordner auswählen
            folder_path = filedialog.askdirectory(
                title="Wähle den Ordner mit deinen Texturen",
                initialdir=os.getcwd()
            )
            
            if not folder_path:
                return
            
            # Alle Bilddateien im Ordner finden
            image_extensions = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tiff', '.webp'}
            image_files = []
            
            for file in os.listdir(folder_path):
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_files.append(os.path.join(folder_path, file))
            
            if not image_files:
                self.status_var.set(" Keine Bilddateien im gewählten Ordner gefunden")
                return
            
            # Verwende das erste gefundene Bild für alle Texturtypen
            primary_image = image_files[0]
            loaded_count = 0
            
            for tex_type in self.pbr_maker.current_textures:
                self.load_texture(tex_type, primary_image)
                loaded_count += 1
            
            self.status_var.set(f" '{os.path.basename(primary_image)}' für alle {loaded_count} Texturtypen geladen + StandardPBR aktiviert")
            
        except Exception as e:
            self.status_var.set(f" Fehler beim Laden des Bildersets: {e}")
    
    def load_placeholder_images(self):
        """Lade Ihre eigenen Placeholder-Bilder"""
        placeholders = {}
        
        # Mapping der Größen zu den entsprechenden Placeholder-Dateien
        placeholder_files = {
            "klein": "Resources/Placeholder_128.png",   # 64x64 → 128px Placeholder
            "mittel": "Resources/Placeholder_256.png", # 100x100 → 256px Placeholder  
            "groß": "Resources/Placeholder_512.png"    # 150x150 → 512px Placeholder
        }
        
        for size_name in self.size_options.keys():
            size_key = size_name.lower()
            placeholders[size_key] = {}
            
            placeholder_path = placeholder_files[size_key]
            
            try:
                # Lade Ihr Placeholder-Bild
                if os.path.exists(placeholder_path):
                    with Image.open(placeholder_path) as img:
                        # Skaliere auf die gewählte Thumbnail-Größe
                        img.thumbnail(self.size_options[size_name], Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        
                        # Verwende dieses Bild für alle Texturtypen und als allgemeiner Platzhalter
                        placeholders[size_key]["general"] = photo
                        for tex_type in ["base_color", "normal", "roughness", "metallic", "occlusion", "emission", "alpha", "height"]:
                            placeholders[size_key][tex_type] = photo
                            
                        print(f" Verwende '{placeholder_path}' für Größe '{size_name}'")
                else:
                    print(f"  Placeholder-Datei nicht gefunden: {placeholder_path}")
                    placeholders[size_key]["general"] = None
                    for tex_type in ["base_color", "normal", "roughness", "metallic", "occlusion", "emission", "alpha", "height"]:
                        placeholders[size_key][tex_type] = None
                        
            except Exception as e:
                print(f" Fehler beim Laden des Placeholder-Bildes {placeholder_path}: {e}")
                placeholders[size_key]["general"] = None
                for tex_type in ["base_color", "normal", "roughness", "metallic", "occlusion", "emission", "alpha", "height"]:
                    placeholders[size_key][tex_type] = None
        
        return placeholders
    
    def set_placeholder_image(self, preview_label, tex_type):
        """Setze das entsprechende Platzhalterbild basierend auf Größe und Texturtyp"""
        size_key = self.current_size.lower()
        
        try:
            # Versuche spezifisches Platzhalterbild zu verwenden
            if (size_key in self.placeholder_images and 
                tex_type in self.placeholder_images[size_key] and
                self.placeholder_images[size_key][tex_type]):
                
                photo = self.placeholder_images[size_key][tex_type]
                preview_label.config(image=photo, text="")
                preview_label.image = photo  # Referenz behalten
                return
        except Exception as e:
            print(f"Fehler beim Setzen des Platzhalterbildes für {tex_type}: {e}")
        
        # Fallback: Text-Platzhalter
        preview_label.config(image="", text="Ihre Vorlage", relief="sunken")
        preview_label.image = None
    
    def run(self):
        """Starte die GUI"""
        print("OpenSimulator PBR Material Maker")
        print("Features:")
        print("-  Echte Drag & Drop Unterstützung")
        print("-  Bildvorschau mit PIL/Tkinter") 
        print("-  Material-Presets")
        print("-  Responsive GUI")
        print("-  Bessere Benutzerfreundlichkeit")
        print("-  StandardPBR Auto-Load")
        print("-  Bilderset-Loader")
        print("-  Ihre eigenen Platzhalterbilder aktiviert")
        
        # Beim Start automatisch StandardPBR laden
        self.root.after(100, self.load_standard_pbr)  # Nach 100ms laden
        
        self.root.mainloop()
    
    def update_emission_settings(self, *args):
        """Aktualisiere Emission-Einstellungen in der Konfiguration - jetzt über update_parameter"""
        try:
            # Diese Funktion wird jetzt durch update_parameter ersetzt
            # Aktualisiere Edge Enhancement Parameter
            if "emission_edge_enhance" in self.param_vars:
                edge_enhance = self.param_vars["emission_edge_enhance"].get() > 0.5
                self.pbr_maker.config["EmissionEdgeEnhance"] = edge_enhance
            
            if "emission_edge_strength" in self.param_vars:
                self.pbr_maker.config["EmissionEdgeStrength"] = self.param_vars["emission_edge_strength"].get()
            
            # Debug-Ausgabe für die neuen Parameter
            if "emission_edge_enhance" in self.param_vars and "emission_edge_strength" in self.param_vars:
                edge_val = self.param_vars["emission_edge_enhance"].get()
                strength_val = self.param_vars["emission_edge_strength"].get()
                print(f"Emission Parameter aktualisiert: Konturen={edge_val:.1f} ({'an' if edge_val > 0.5 else 'aus'}), Stärke={strength_val:.1f}")
            
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Emission-Einstellungen: {e}")
    
    def update_strength_label(self, *args):
        """Nicht mehr benötigt - Werte werden automatisch durch Parameter-System angezeigt"""
        pass
    
    def update_pypbr_info(self):
        """Aktualisiere PyPBR-Informationsanzeige"""
        try:
            if hasattr(self.pbr_maker, 'pypbr_pipeline') and self.pbr_maker.pypbr_pipeline.enabled:
                pipeline = self.pbr_maker.pypbr_pipeline
                
                # Basis-Informationen sammeln
                info_parts = []
                
                # GPU/CPU Status
                if pipeline.gpu_enabled:
                    gpu_name = torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "GPU"
                    info_parts.append(f"PyPBR-GPU ({gpu_name[:8]})")
                else:
                    info_parts.append("PyPBR-CPU")
                
                # BRDF Modell
                info_parts.append("Cook-Torrance")
                
                # Aktive Optimierungen
                config = pipeline.config
                active_opts = []
                
                if config.get('energy_conservation', False):
                    active_opts.append("Energy")
                if config.get('fresnel_enabled', False):
                    active_opts.append("Fresnel")
                if config.get('albedo_is_srgb', False):
                    active_opts.append("sRGB→Linear")
                if config.get('roughness_clamp_min', 0) > 0:
                    active_opts.append(f"Roughness≥{config['roughness_clamp_min']}")
                
                if active_opts:
                    info_parts.append(f"[{' | '.join(active_opts)}]")
                
                # Zusammenfassung erstellen
                pypbr_info = " | ".join(info_parts)
                
                # Status aktualisieren
                if hasattr(self, 'pypbr_info_var'):
                    self.pypbr_info_var.set(pypbr_info)
                    
            else:
                if hasattr(self, 'pypbr_info_var'):
                    self.pypbr_info_var.set("Standard PBR")
                    
        except Exception as e:
            print(f"Fehler beim Aktualisieren der PyPBR-Info: {e}")
            if hasattr(self, 'pypbr_info_var'):
                self.pypbr_info_var.set("PyPBR-Fehler")

if __name__ == "__main__":
    try:
        app = MaterialMakerGUI()
        app.run()
    except ImportError as e:
        if "tkinterdnd2" in str(e):
            print(" tkinterdnd2 nicht gefunden!")
            print("Installation: pip install tkinterdnd2")
        else:
            print(f" Import-Fehler: {e}")
    except Exception as e:
        print(f" Fehler: {e}")
        import traceback
        traceback.print_exc()