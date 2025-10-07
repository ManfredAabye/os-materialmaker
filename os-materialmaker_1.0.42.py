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
    print("✅ GLTF + PyRender verfügbar - 3D-Rendering aktiviert")
except ImportError as e:
    GLTF_AVAILABLE = False
    PYRENDER_AVAILABLE = False
    # Fallback-Module auf None setzen
    pyrender = None
    trimesh = None
    GLTF2 = None
    print(f"GLTF/PyRender-Bibliotheken nicht verfügbar: {e}. Fallback auf einfache 3D-Darstellung.")

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
                print("🔧 Initialisiere PyRender...")
                
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
                    print("✅ PyRender OffscreenRenderer erfolgreich initialisiert")
                    
                except Exception as renderer_error:
                    print(f"⚠️ OffscreenRenderer Fehler: {renderer_error}")
                    print("🔄 Versuche alternative Renderer-Konfiguration...")
                    
                    try:
                        # Fallback: Mesa Software-Rendering
                        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
                        self.renderer = pyrender.OffscreenRenderer(
                            viewport_width=self.width, 
                            viewport_height=self.height
                        )
                        print("✅ PyRender mit Mesa Software-Rendering initialisiert")
                        
                    except Exception as mesa_error:
                        print(f"⚠️ Mesa Renderer Fehler: {mesa_error}")
                        print("🔄 Verwende einfachen Fallback-Renderer...")
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
                    
                    print("✅ PyRender Beleuchtung konfiguriert")
                else:
                    print("⚠️ PyRender nicht verfügbar - verwende Fallback")
                    self.use_fallback_renderer = True
            else:
                print("⚠️ PyRender nicht verfügbar")
                self.renderer = None
                self.use_fallback_renderer = True
                
        except Exception as e:
            print(f"❌ Kritischer Fehler bei PyRender-Initialisierung: {e}")
            self.renderer = None
            self.scene = None
            self.use_fallback_renderer = True
    
    def load_models(self):
        """3D-Modelle nicht mehr erforderlich - verwende nur noch 2D-Darstellung"""
        print("ℹ️ 3D-Modelle übersprungen - verwende nur noch 2D-Fallback-Darstellung")
        self.current_model = 'fallback'
    
    def update_gltf_texture(self, model_type, texture_type, image_data):
        """GLTF2-Funktionalität deaktiviert - verwendet nur noch Fallback-Renderer"""
        #print(f"ℹ️ GLTF2-Objekte deaktiviert - Verwende Fallback-Renderer für Textur-Darstellung")
        return False
    
    def export_gltf_model(self, filename="exported_model.glb"):
        """GLTF2-Export deaktiviert - nur Fallback-Renderer verfügbar"""
        #print("ℹ️ GLTF2-Export deaktiviert - Verwende Fallback-Renderer für Textur-Darstellung")
        return False
    
    def create_fallback_models(self):
        """Fallback-Modelle sind nicht mehr erforderlich - nur noch 2D-Darstellung"""
        print("ℹ️ Verwende nur noch 2D-Fallback-Darstellung - keine 3D-Modelle erforderlich")
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
        print(f"🎨 Set Textures aufgerufen mit {len(texture_images) if texture_images else 0} Texturen")
        if texture_images:
            for tex_type, img in texture_images.items():
                print(f"  - {tex_type}: {img.size if hasattr(img, 'size') else 'Unbekannt'}")
                
                # 📝 Aktualisiere GLTF2-Objekt mit neuer Textur
                self.update_gltf_texture(self.current_model, tex_type, img)
        
        self.texture_images = texture_images or {}
        print("🔄 Starte Render-Vorgang...")
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
            print(f"❌ PyRender Fehler: {render_error}")
            print("🔄 Wechsle zu Fallback-Renderer...")
            self.use_fallback_renderer = True
            return self.render_fallback()
    
    def render_with_pyrender(self):
        """Sicheres PyRender-Rendering mit verbesserter Fehlerbehandlung"""
        try:
            # 🧹 VOLLSTÄNDIGE Scene-Bereinigung
            # Entferne alle bestehenden Nodes UND bereinige Renderer
            if self.scene is not None and hasattr(self.scene, 'mesh_nodes') and hasattr(self.scene, 'camera_nodes'):
                nodes_to_remove = list(self.scene.mesh_nodes) + list(self.scene.camera_nodes)
                for node in nodes_to_remove:
                    try:
                        self.scene.remove_node(node)
                    except Exception:
                        pass  # Ignoriere Fehler beim Entfernen
            
            # 🔄 Erstelle KOMPLETT NEUES MESH (nie wiederverwenden)
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
                    
                    print("✅ Isoliertes Mesh mit Textur-Material erstellt")
                except Exception as mesh_error:
                    print(f"⚠️ Fehler beim Erstellen des isolierten Mesh: {mesh_error}")
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
                            print("✅ Fallback-Mesh aus Trimesh erstellt")
                        else:
                            mesh = original_mesh
                            print("⚠️ Verwende Original-Mesh")
                    except Exception:
                        mesh = original_mesh
                        print("⚠️ Verwende Original-Mesh")
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
                help_text = "Maus: Rotation | Mausrad: Zoom | PyRender aktiv"
                self.canvas.create_text(
                    10, self.height-10, text=help_text, fill="lime", font=("Arial", 9), anchor="sw"
                )
                
            except Exception as render_error:
                print(f"❌ Rendering-Fehler: {render_error}")
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
            
            # 🧹 CLEANUP nach Rendering
            try:
                # Entferne Kamera-Node
                if camera_node is not None and self.scene is not None:
                    self.scene.remove_node(camera_node)
            except Exception:
                pass  # Ignoriere Cleanup-Fehler
                
        except Exception as e:
            print(f"⚠️ Fehler beim PyRender-Rendering: {e}")
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
            help_text = "Maus: Rotation | Mausrad: Zoom | Plane-Renderer aktiv"
            self.canvas.create_text(
                10, self.height-10, text=help_text, fill="yellow", font=("Arial", 9), anchor="sw"
            )
            
        except Exception as e:
            print(f"❌ Fallback-Render-Fehler: {e}")
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
            print(f"⚠️ Fehler beim Textur-Overlay: {e}")
    
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
            print(f"⚠️ Fehler bei PBR-Kombination: {e}")
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
            print(f"⚠️ Fehler bei PBR-Indikatoren: {e}")
    
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
        """Zeichne Textur-Overlay auf der statischen Plane"""
        try:
            if 'base_color' in self.texture_images:
                base_texture = self.texture_images['base_color']
                size = min(self.width, self.height) * 0.25 * self.zoom
                
                # Kombiniere PBR-Texturen
                combined_texture = self.combine_pbr_textures(base_texture, int(size))
                
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
            print(f"⚠️ Fehler beim Plane-Textur-Overlay: {e}")
    
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
            print(f"⚠️ Fehler bei PBR-Farbkombination: {e}")
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
                print("⚠️ PyRender nicht verfügbar für Material-Erstellung")
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
                    print(f"✅ Base Color Textur erstellt: {base_array.shape}")
                    
                except Exception as tex_error:
                    print(f"⚠️ Fehler beim Laden der Base Color Textur: {tex_error}")
            
            # Normal Map hinzufügen (wenn vorhanden)
            if 'normal' in self.texture_images:
                try:
                    normal_img = self.texture_images['normal']
                    if normal_img.mode != 'RGB':
                        normal_img = normal_img.convert('RGB')
                    
                    normal_array = np.array(normal_img, dtype=np.uint8)
                    normal_texture = pyrender.Texture(source=normal_array, source_channels='RGB')
                    material_params['normalTexture'] = normal_texture
                    print(f"✅ Normal Map Textur erstellt: {normal_array.shape}")
                    
                except Exception as tex_error:
                    print(f"⚠️ Fehler beim Laden der Normal Map: {tex_error}")
            
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
                        print("✅ Metallic/Roughness Textur kombiniert")
                        
                except Exception as tex_error:
                    print(f"⚠️ Fehler beim Erstellen der Metallic/Roughness Textur: {tex_error}")
            
            # Erstelle PyRender Material
            if pyrender is not None:
                material = pyrender.MetallicRoughnessMaterial(**material_params)
                return material
            else:
                print("⚠️ PyRender nicht verfügbar")
                return None
                
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen des Materials: {e}")
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

class GLTFViewer:
    """Vereinfachter GLTF-Viewer für Tkinter Canvas"""
    def __init__(self):
        self.gltf_models = {}
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
                    print("✅ Cube.gltf geladen")
                
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
                    print("✅ Ball.gltf geladen")
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der GLTF-Modelle: {e}")
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
            print(f"⚠️ Fehler beim GLTF-Rendering: {e}")
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
        
        # Standard-Konfiguration
        self.config = {
            "NormalStrength": 0.20,
            "RoughnessStrength": 0.20,
            "OcclusionStrength": 1.0,
            "MetallicThreshold": 127,
            "EmissionStrength": 0.0,
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
            "base_color": ["_albedo", "_diffuse", "_color", "_basecolor", "_base_color", "_diff", "_col"],
            "normal": ["_normal", "_norm", "_nrm", "_normalmap", "_normal_map"],
            "roughness": ["_roughness", "_rough", "_rgh", "_roughnessmap"],
            "metallic": ["_metallic", "_metal", "_met", "_metallicmap"],
            "occlusion": ["_ao", "_occlusion", "_ambient", "_ambientocclusion", "_ambient_occlusion"],
            "emission": ["_emission", "_emissive", "_emit", "_glow", "_light"],
            "alpha": ["_alpha", "_opacity", "_transparent", "_mask"],
            "height": ["_height", "_displacement", "_disp", "_bump"]
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
            
            # Lade alle Materialien
            for name, material in data.items():
                presets[name] = material
            
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
        
        self.root.title(f"OpenSimulator PBR Material Maker v{version} - GLTF Edition")
        self.root.geometry("2400x1800")  # Weiter vergrößert für 1200x1200 GLTF-Vorschau + komplettes UI
        
        # PBR Material Maker Backend
        self.pbr_maker = PBRMaterialMaker()
        
        # GLTF Viewer für 3D-Vorschau (wird nach Canvas-Erstellung initialisiert)
        self.gltf_viewer = GLTFViewer()
        self.interactive_3d_viewer = None  # Wird später initialisiert
        
        # Bild-Cache für Vorschaubilder
        self.preview_images = {}
        
        # Aktuelle Texturen für GLTF-Rendering
        self.current_texture_images = {}
        
        # Größenoptionen für Thumbnails und Platzhalter
        self.size_options = {
            "Klein": (64, 64),
            "Mittel": (100, 100), 
            "Groß": (150, 150)
        }
        self.current_size = "Mittel"
        self.thumbnail_size = self.size_options[self.current_size]
        
        # Status-Variable
        self.status_var = tk.StringVar()
        self.status_var.set("Bereit")
        
        # Preview Mode für 3D-Viewer
        self.preview_mode = tk.StringVar(value="cube")
        
        # Texture Labels Dict
        self.texture_labels = {}
        self.texture_frames = {}
        
        # Lade Platzhalterbilder für alle Größen
        self.placeholder_images = self.load_placeholder_images()
        
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
        title_label = ttk.Label(main_frame, text="📦 PBR Material Maker - Tkinter Edition", 
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
        texture_frame = ttk.LabelFrame(parent, text="🖼️ Texturen", padding="10")
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
            
            # Drag & Drop Support
            preview_label.drop_target_register(DND_FILES)  # type: ignore
            preview_label.dnd_bind('<<Drop>>', lambda e, t=tex_type: self.on_drop(e, t))  # type: ignore
            
            # Click-Handler
            preview_label.bind("<Button-1>", lambda e, t=tex_type: self.select_texture_file(t))
            
            # Dateiname-Label (sehr kompakt)
            file_label = ttk.Label(tex_frame, text="Keine Datei", foreground="gray", font=("Arial", 8))
            file_label.grid(row=2, column=0)
            
            self.texture_frames[tex_type] = tex_frame
            self.texture_labels[tex_type] = (preview_label, file_label)
        
        # Konfiguriere Grid-Gewichte für gleichmäßige Verteilung
        for i in range(4):
            texture_frame.columnconfigure(i, weight=1)
        
        # Action Buttons kompakt unten
        button_frame = ttk.Frame(texture_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="🔍 Auto-Find", 
                  command=self.auto_find_textures).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="✨ Maps generieren", 
                  command=self.generate_missing_maps).grid(row=0, column=1, padx=2)
        ttk.Button(button_frame, text="🧹 Clear All", 
                  command=self.clear_all_textures).grid(row=0, column=2, padx=2)
        ttk.Button(button_frame, text="📁 Bilderset laden", 
                  command=self.load_image_set).grid(row=0, column=3, padx=2)
        
    def setup_config_panel_compact(self, parent):
        """Erstelle das kompakte Konfigurations-Panel"""
        config_frame = ttk.LabelFrame(parent, text="⚙️ Konfiguration", padding="10")
        config_frame.grid(row=1, column=0, sticky="nsew")
        
        # Preset-Auswahl
        preset_frame = ttk.LabelFrame(config_frame, text="📋 Material Presets", padding="5")
        preset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        ttk.Label(preset_frame, text="Preset auswählen:").grid(row=0, column=0, sticky="w")
        
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                  values=list(self.pbr_maker.material_presets.keys()),
                                  state="readonly")
        preset_combo.grid(row=1, column=0, sticky="ew", pady=2)
        preset_combo.bind('<<ComboboxSelected>>', self.apply_preset)
        
        # Größeneinstellungen kompakt
        size_frame = ttk.LabelFrame(config_frame, text="📏 Größe", padding="5")
        size_frame.grid(row=1, column=0, sticky="ew", pady=2)
        
        ttk.Label(size_frame, text="Größe:").grid(row=0, column=0, sticky="w")
        
        self.size_var = tk.StringVar(value=self.current_size)
        size_combo = ttk.Combobox(size_frame, textvariable=self.size_var,
                                values=list(self.size_options.keys()), 
                                state="readonly", width=10)
        size_combo.grid(row=1, column=0, sticky="ew", pady=2)
        size_combo.bind('<<ComboboxSelected>>', self.change_thumbnail_size)
        
        # Größen-Info
        self.size_info_var = tk.StringVar()
        self.update_size_info()
        size_info_label = ttk.Label(size_frame, textvariable=self.size_info_var, foreground="blue", font=("Arial", 8))
        size_info_label.grid(row=2, column=0, sticky="w")
        
        # Parameter-Einstellungen kompakt
        param_frame = ttk.LabelFrame(config_frame, text="🎚️ Parameter", padding="5")
        param_frame.grid(row=2, column=0, sticky="ew", pady=2)
        
        self.param_vars = {}
        parameters = [
            ("normal_strength", "Normal Stärke", 0.0, 2.0, 0.2),
            ("roughness_strength", "Roughness Stärke", 0.0, 2.0, 0.2),
            ("occlusion_strength", "AO Stärke", 0.0, 2.0, 1.0),
            ("metallic_threshold", "Metallic Schwelle", 0, 255, 127),
            ("emission_strength", "Emission Stärke", 0.0, 2.0, 0.0),
            ("alpha_strength", "Alpha Stärke", 0.0, 2.0, 1.0)
        ]
        
        for i, (param_name, display_name, min_val, max_val, default) in enumerate(parameters):
            ttk.Label(param_frame, text=f"{display_name}:", font=("Arial", 8)).grid(row=i, column=0, sticky="w")
            
            if isinstance(default, float):
                var = tk.DoubleVar()
                var.set(default)
            else:
                var = tk.IntVar()
                var.set(int(default))
            self.param_vars[param_name] = var
            
            scale = ttk.Scale(param_frame, from_=min_val, to=max_val, variable=var, 
                            orient=tk.HORIZONTAL, length=120)
            scale.grid(row=i, column=1, sticky="ew", padx=5, pady=1)
            scale.bind('<Motion>', lambda e, p=param_name: self.update_parameter(p))
            
            # Wert-Anzeige
            value_label = ttk.Label(param_frame, textvariable=var, font=("Arial", 8))
            value_label.grid(row=i, column=2, padx=5)
        
        # Export Buttons kompakt
        export_frame = ttk.Frame(config_frame)
        export_frame.grid(row=3, column=0, pady=10)
        
        ttk.Button(export_frame, text="💾 GLTF Export", 
                  command=self.export_gltf).grid(row=0, column=0, padx=2)
        ttk.Button(export_frame, text="📸 Screenshot", 
                  command=self.export_screenshot).grid(row=0, column=1, padx=2)
        
        # Export-Optionen (vereinfacht)
        export_options_frame = ttk.LabelFrame(config_frame, text="📤 Export-Optionen", padding="5")
        export_options_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        
        # Export-Buttons
        export_buttons_frame = ttk.Frame(export_options_frame)
        export_buttons_frame.grid(row=0, column=0, sticky="ew")
        
        ttk.Button(export_buttons_frame, text="📸 Screenshot", 
                  command=self.export_screenshot).grid(row=0, column=0, padx=(0, 3))
        ttk.Button(export_buttons_frame, text="💾 Material JSON", 
                  command=self.export_gltf).grid(row=0, column=1, padx=(0, 3))
        ttk.Button(export_buttons_frame, text="📦 Save GLTF", 
                  command=self.save_gltf_for_secondlife).grid(row=0, column=2)
        
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
                
                # ✅ FIX: Setze pbr_maker.current_textures auf None zurück (nicht löschen!)
                self.pbr_maker.current_textures[tex_type] = None
            
            # Aktualisiere 3D-Viewer
            if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                self.interactive_3d_viewer.set_textures({})
            
            self.status_var.set("🧹 Alle Texturen gelöscht - Placeholder wiederhergestellt")
            print("✅ Alle Texturen erfolgreich gelöscht und Placeholder-Bilder wiederhergestellt")
            
        except Exception as e:
            print(f"❌ Fehler beim Löschen der Texturen: {e}")
            self.status_var.set("❌ Fehler beim Löschen")
        
    def load_image_set(self):
        """Lade ein komplettes Bilderset"""
        # TODO: Implementiere Bilderset-Loader
        self.status_var.set("📁 Bilderset laden - TODO")
        
    def update_parameter(self, param_name):
        """Aktualisiere Parameter und GLTF-Vorschau"""
        try:
            # Parameter-Wert abrufen
            if param_name in self.param_vars:
                value = self.param_vars[param_name].get()
                print(f"🔄 Parameter '{param_name}' geändert: {value}")
                
                # 🔄 Automatisches GLTF-Update nach Parameter-Änderung
                self.refresh_gltf_preview()
                
        except Exception as e:
            print(f"❌ Fehler beim Parameter-Update '{param_name}': {e}")
        
        # Status-Bar über alle Spalten
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0)
        status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="green")
        status_label.grid(row=0, column=1, sticky="w", padx=(10, 0))
        
    def setup_texture_panel(self, parent):
        """Erstelle das Textur-Panel"""
        texture_frame = ttk.LabelFrame(parent, text="🖼️ Texturen", padding="10")
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
            
            # Dateiname-Label
            file_label = ttk.Label(tex_frame, text="Keine Datei", foreground="gray")
            file_label.grid(row=2, column=0, columnspan=2)
            
            self.texture_frames[tex_type] = tex_frame
            self.texture_labels[tex_type] = (preview_label, file_label)
        
        # Action Buttons
        button_frame = ttk.Frame(texture_frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=20)
        
        # Erste Reihe
        ttk.Button(button_frame, text="🔍 Auto-Find", 
                  command=self.auto_find_textures).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="✨ Maps generieren", 
                  command=self.generate_missing_maps).grid(row=0, column=1, padx=5)
        
        # Zweite Reihe
        ttk.Button(button_frame, text="🖼️ Ein Bild für alle", 
                  command=self.load_user_imageset).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="🎯 StandardPBR laden", 
                  command=self.load_standard_pbr).grid(row=1, column=1, padx=5, pady=5)
        
        # Dritte Reihe
        ttk.Button(button_frame, text="💾 Material-Paket", 
                  command=self.save_material_package).grid(row=2, column=0, padx=5)
        ttk.Button(button_frame, text="🧹 Alles löschen", 
                  command=self.clear_all).grid(row=2, column=1, padx=5)
    
    def setup_config_panel(self, parent):
        """Erstelle das Konfigurations-Panel"""
        config_frame = ttk.LabelFrame(parent, text="⚙️ Konfiguration", padding="10")
        config_frame.grid(row=1, column=1, sticky="nswe")
        
        # Material-Presets
        preset_frame = ttk.LabelFrame(config_frame, text="🎨 Material-Presets", padding="10")
        preset_frame.grid(row=0, column=0, sticky="we", pady=(0, 10))
        
        ttk.Label(preset_frame, text="Preset auswählen:").grid(row=0, column=0, sticky=tk.W)
        
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                  values=list(self.pbr_maker.material_presets.keys()),
                                  state="readonly")
        preset_combo.grid(row=1, column=0, sticky="we", pady=5)
        preset_combo.bind('<<ComboboxSelected>>', self.apply_preset)
        
        # Größeneinstellungen für Vorschaubilder
        size_frame = ttk.LabelFrame(config_frame, text="📏 Vorschaubild-Größe", padding="10")
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
        param_frame = ttk.LabelFrame(config_frame, text="🎚️ Parameter", padding="10")
        param_frame.grid(row=2, column=0, sticky="we", pady=(0, 10))
        
        self.param_vars = {}
        parameters = [
            ("normal_strength", "Normal Stärke", 0.0, 2.0, 0.2),
            ("roughness_strength", "Roughness Stärke", 0.0, 2.0, 0.2),
            ("occlusion_strength", "AO Stärke", 0.0, 2.0, 1.0),
            ("metallic_threshold", "Metallic Schwelle", 0, 255, 127),
            ("emission_strength", "Emission Stärke", 0.0, 2.0, 0.0),
            ("alpha_strength", "Alpha Stärke", 0.0, 2.0, 1.0)
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(parameters):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            
            if isinstance(default, float):
                var = tk.DoubleVar()
                var.set(default)
            else:
                var = tk.IntVar()
                var.set(int(default))
            self.param_vars[key] = var
            
            scale = ttk.Scale(param_frame, from_=min_val, to=max_val, variable=var, 
                            orient=tk.HORIZONTAL, length=200)
            scale.grid(row=i, column=1, sticky="we", padx=10, pady=2)
            
            value_label = ttk.Label(param_frame, text=f"{default:.2f}" if isinstance(default, float) else str(default))
            value_label.grid(row=i, column=2, pady=2)
            
            # Update-Handler mit Auto-Refresh
            var.trace('w', lambda *args, lbl=value_label, v=var, param=key: self.update_param_with_refresh(lbl, v, param))
    
    def update_param_with_refresh(self, label, var, param_name):
        """Update parameter label and trigger GLTF refresh"""
        try:
            # Aktualisiere Label
            value = var.get()
            if isinstance(value, float):
                label.config(text=f"{value:.2f}")
            else:
                label.config(text=str(int(value)))
            
            print(f"🔄 Parameter '{param_name}' geändert: {value}")
            
            # Automatisches GLTF-Update nach Parameter-Änderung
            self.refresh_gltf_preview()
            
        except Exception as e:
            print(f"❌ Fehler beim Parameter-Update '{param_name}': {e}")
    
    
    def setup_preview_panel(self, parent):
        """Erstelle das vergrößerte Textur-Plane Vorschau Panel"""
        preview_frame = ttk.LabelFrame(parent, text="� Textur-Plane Vorschau", padding="15")
        preview_frame.grid(row=1, column=1, sticky="nsew")
        
        # Info-Text
        info_frame = ttk.Frame(preview_frame)
        info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        ttk.Label(info_frame, text="Interaktive Textur-Plane:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, text="• Maus ziehen: Rotation", font=("Arial", 9)).grid(row=1, column=0, sticky="w")
        ttk.Label(info_frame, text="• Mausrad: Zoom", font=("Arial", 9)).grid(row=2, column=0, sticky="w")
        
        # 3D-Vorschau Bereich (200% vergrößert: 600x600 -> 1200x1200)
        self.preview_canvas = tk.Canvas(preview_frame, width=1200, height=1200, bg="black", relief="sunken", bd=2)
        self.preview_canvas.grid(row=1, column=0, pady=15)
        
        # Initialisiere PyRender-basierten 3D-Viewer (nur Plane)
        self.interactive_3d_viewer = PyRender3DViewer(self.preview_canvas, 1200, 1200)
        
        # Setze Standard-Modell auf Plane
        if hasattr(self.interactive_3d_viewer, 'current_model'):
            self.interactive_3d_viewer.current_model = 'plane'
    
    def create_placeholder_3d(self):
        """Erstelle GLTF-basierte 3D-Szene mit echten 3D-Modellen"""
        self.preview_canvas.delete("all")
        
        # Canvas-Dimensionen (600x600)
        canvas_width = 600
        canvas_height = 600
        
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
            print(f"⚠️ Fehler beim GLTF-Rendering: {e}")
        
        # Fallback auf einfache Darstellung
        self.create_fallback_3d()
    
    def create_fallback_3d(self):
        """Erstelle Fallback 3D-Darstellung wenn GLTF nicht funktioniert"""
        canvas_width = 600
        canvas_height = 600
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
        """Erstelle einfache Kugel mit Schattierung (vergrößerte Version)"""
        base_radius = 120  # Vergrößert von 60 auf 120
        
        for i in range(8):  # Mehr Schichten für besseren Effekt
            shade = 80 - i * 10
            color = f"#{shade:02x}{shade:02x}{shade:02x}"
            radius = base_radius - i * 12
            
            self.preview_canvas.create_oval(
                cx-radius, cy-radius, cx+radius, cy+radius,
                fill=color, outline="", tags="object"
            )
        
        # Highlight (vergrößert)
        highlight_size = 35
        self.preview_canvas.create_oval(
            cx-highlight_size*2, cy-highlight_size*2, cx-highlight_size//2, cy-highlight_size//2,
            fill="#ffffff", outline="", tags="object"
        )
    
    def update_preview_mode(self):
        """Plane-Modus (nicht mehr verwendet da nur noch Plane)"""
        self.status_var.set("� Textur-Plane aktiv")
    
    def update_lighting(self, event=None):
        """Beleuchtung nicht mehr verwendet (nur noch 2D-Darstellung)"""
        self.status_var.set("ℹ️ Beleuchtung nicht verfügbar - 2D-Darstellung aktiv")
    
    def update_renderer_mode(self, event=None):
        """Renderer-Modus nicht mehr verwendet (nur noch 2D-Darstellung)"""
        self.status_var.set("ℹ️ Verwendet automatisch 2D-Fallback-Darstellung")
    
    def update_rotation(self, event=None):
        """Rotation nicht mehr verwendet (statische 2D-Darstellung)"""
        self.status_var.set("ℹ️ Rotation deaktiviert - statische Darstellung aktiv")
    
    def refresh_gltf_preview(self):
        """Zentrale Methode für automatisches GLTF-Vorschau-Update"""
        try:
            if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                print("🔄 Automatisches GLTF-Vorschau-Update...")
                
                # Sammle alle aktuellen Texturen
                texture_images = {}
                
                # 1. Zuerst bereits geladene Images verwenden (z.B. generierte Maps)
                if hasattr(self, 'current_texture_images'):
                    for texture_type, texture_image in self.current_texture_images.items():
                        if texture_image is not None:
                            texture_images[texture_type] = texture_image
                            print(f"  ✅ Textur '{texture_type}' aus Cache geladen")
                
                # 2. Dann fehlende Texturen aus Pfaden laden
                for texture_type, texture_path in self.pbr_maker.current_textures.items():
                    if texture_type not in texture_images and texture_path and os.path.exists(texture_path):
                        try:
                            texture_image = Image.open(texture_path)
                            texture_images[texture_type] = texture_image
                            print(f"  ✅ Textur '{texture_type}' aus Datei geladen")
                        except Exception as e:
                            print(f"  ⚠️ Fehler beim Laden von '{texture_type}': {e}")
                
                # Aktualisiere 3D-Viewer
                self.interactive_3d_viewer.set_textures(texture_images)
                print("✅ GLTF-Vorschau automatisch aktualisiert")
                
            else:
                print("⚠️ Kein 3D-Viewer für Auto-Update verfügbar")
                
        except Exception as e:
            print(f"❌ Fehler beim automatischen GLTF-Update: {e}")
    
    def update_texture_for_gltf(self, texture_type):
        """Aktualisiere Textur für GLTF-Rendering"""
        try:
            # Hole aktuellen Textur-Pfad
            texture_path = self.pbr_maker.current_textures.get(texture_type)
            print(f"🔄 Update GLTF Textur '{texture_type}': {texture_path}")
            
            if texture_path and os.path.exists(texture_path):
                # Lade PIL Image für GLTF-Rendering
                texture_image = Image.open(texture_path)
                self.current_texture_images[texture_type] = texture_image
                print(f"✅ Textur '{texture_type}' geladen: {texture_image.size}")
                
                # Aktualisiere 3D-Viewer mit neuen Texturen
                if hasattr(self, 'interactive_3d_viewer') and self.interactive_3d_viewer:
                    print("🔄 Aktualisiere 3D-Viewer mit neuer Textur...")
                    self.interactive_3d_viewer.set_textures(self.current_texture_images)
                    print("✅ 3D-Viewer aktualisiert")
                else:
                    print("⚠️ Kein 3D-Viewer verfügbar")
            else:
                print(f"❌ Textur-Pfad nicht gefunden: {texture_path}")
                
        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren der GLTF-Textur {texture_type}: {e}")
    
    def on_drop(self, event, texture_type):
        """Handle Drag & Drop von Dateien"""
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            self.load_texture(texture_type, file_path)
            self.status_var.set(f"✅ {texture_type.upper()} via Drag & Drop geladen")
    
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
            # Setze Textur-Pfad
            self.pbr_maker.current_textures[texture_type] = file_path
            
            # Lade Vorschaubild
            with Image.open(file_path) as img:
                # Erstelle Thumbnail
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Aktualisiere UI
                preview_label, file_label = self.texture_labels[texture_type]
                preview_label.config(image=photo, text="")
                preview_label.image = photo  # Referenz behalten
                
                file_label.config(text=self.truncate_filename(os.path.basename(file_path)), foreground="black")
                
                self.status_var.set(f"✅ {texture_type.upper()} geladen: {os.path.basename(file_path)}")
                
                # 🔄 Automatisches GLTF-Update nach Textur-Laden
                self.refresh_gltf_preview()
                
        except Exception as e:
            self.status_var.set(f"❌ Fehler beim Laden: {str(e)}")
    
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
        self.status_var.set("ℹ️ Auto-Rotation deaktiviert - statische Darstellung aktiv")
    
    def export_screenshot(self):
        """Exportiere Screenshot der 3D-Vorschau"""
        self.status_var.set("📸 Screenshot wird erstellt...")
        # TODO: Implementiere Screenshot-Export
        messagebox.showinfo("Screenshot", "Screenshot-Export wird in einer zukünftigen Version implementiert")
        
    def export_gltf(self):
        """Exportiere GLTF mit aktuellen Einstellungen"""
        self.status_var.set("💾 GLTF-Export wird erstellt...")
        # TODO: Implementiere erweiterten GLTF-Export
        messagebox.showinfo("GLTF Export", "GLTF-Export wird in einer zukünftigen Version implementiert")
    
    def save_gltf_for_secondlife(self):
        """Speichere GLTF und Texturen für Second Life / OpenSim Kompatibilität"""
        try:
            # Prüfe ob Base Color Textur verfügbar ist
            if "base_color" not in self.pbr_maker.current_textures or not self.pbr_maker.current_textures["base_color"]:
                messagebox.showwarning(
                    "Base Color erforderlich", 
                    "Bitte laden Sie zuerst eine Base Color Textur.\n\n"
                    "Die GLTF-Dateien werden im gleichen Verzeichnis wie die Base Color Textur gespeichert."
                )
                return
            
            # Bestimme Ausgabe-Verzeichnis basierend auf Base Color Textur
            base_color_path = self.pbr_maker.current_textures["base_color"]
            base_color_dir = os.path.dirname(base_color_path)
            
            # Erstelle 'gltf_textures' Unterordner im Base Color Verzeichnis
            output_dir = os.path.join(base_color_dir, "gltf_textures")
            
            # Erstelle Verzeichnis falls es nicht existiert
            os.makedirs(output_dir, exist_ok=True)
            
            # Erstelle Basis-Namen basierend auf Base Color Textur
            base_name = os.path.splitext(os.path.basename(base_color_path))[0]
            # Entferne Base Color Suffixe
            for suffix in ["_albedo", "_diffuse", "_color", "_basecolor", "_base_color", "_diff", "_col"]:
                base_name = base_name.replace(suffix, "")
            
            self.status_var.set(f"📦 Erstelle GLTF Export: {base_name}...")
            
            # Sammle alle verfügbaren Texturen
            texture_files = {}
            texture_size = 1024  # Standard-Größe für Second Life/OpenSim
            
            # Exportiere Texturen in Standard-Formaten (mit korrekten Dateinamen)
            for tex_type in ["base_color", "normal", "roughness", "metallic", "occlusion", "emission", "alpha", "height"]:
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
                            print(f"⚠️ Kein gültiger Pfad für {tex_type}: None")
                            continue
                    except Exception as e:
                        print(f"⚠️ Fehler beim Laden von {tex_type}: {e}")
                        continue
                
                if texture_image:
                    # Erstelle Dateinamen entsprechend dem GLTF-Packer Standard
                    tex_mapping = {
                        "base_color": "_col",
                        "normal": "_nrm", 
                        "roughness": "_rough",
                        "metallic": "_metal",
                        "occlusion": "_occ",
                        "emission": "_emission",
                        "alpha": "_alpha",
                        "height": "_height"
                    }
                    
                    texture_filename = f"{base_name}{tex_mapping[tex_type]}.png"
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
                        print(f"✅ Textur gespeichert: {texture_path}")
                    except Exception as e:
                        print(f"❌ Fehler beim Speichern von {tex_type}: {e}")
                else:
                    print(f"⚠️ Keine {tex_type} Textur verfügbar")
            
            print(f"📦 Gespeicherte Texturen: {list(texture_files.values())}")
            
            # Erstelle GLTF-kompatible Material-Definition
            gltf_material, positions, texcoords, indices = self.create_secondlife_gltf_material(base_name, texture_files)
            
            # Speichere GLTF-Datei
            gltf_filename = f"{base_name}.gltf"
            gltf_path = os.path.join(output_dir, gltf_filename)
            
            with open(gltf_path, 'w', encoding='utf-8') as f:
                json.dump(gltf_material, f, indent=2, ensure_ascii=False)
            
            # Erstelle und speichere Binärdatei (.bin)
            bin_filename = f"{base_name}.bin"
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
            
            print(f"✅ GLTF-Datei erstellt: {gltf_path}")
            print(f"✅ Binärdatei erstellt: {bin_path}")
            
            # Erstelle auch eine Info-Datei mit Anweisungen
            info_filename = f"{base_name}_README.txt"
            info_path = os.path.join(output_dir, info_filename)
            
            info_content = f"""Second Life / OpenSim Material Export (GLTF-Packer kompatibel)
==================================================================

Material Name: {base_name}
Export Date: {os.path.basename(__file__)} - {self.__class__.__name__}
Base Color Source: {os.path.basename(base_color_path)}
Export Directory: gltf_textures/

Exported Files:
--------------
- {gltf_filename} (GLTF Material Definition)
- {bin_filename} (GLTF Binary Geometry Data)
{chr(10).join(f"- {filename} ({tex_type.replace('_', ' ').title()} Texture)" for tex_type, filename in texture_files.items())}

Directory Structure:
-------------------
{os.path.basename(base_color_dir)}/
├── (Base Color Textur und andere Original-Texturen)
└── gltf_textures/
    ├── {gltf_filename}
    ├── {bin_filename}
    ├── {base_name}_README.txt
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
            file_count = len(texture_files) + 2  # +2 für GLTF und README
            self.status_var.set(f"✅ GLTF Export abgeschlossen: {file_count} Dateien erstellt")
            
            messagebox.showinfo(
                "GLTF Export erfolgreich", 
                f"Second Life / OpenSim Material exportiert!\n\n"
                f"📁 Verzeichnis: {os.path.relpath(output_dir, base_color_dir)}\n"
                f"📋 Dateien: {file_count} (GLTF + {len(texture_files)} Texturen + README)\n"
                f"📂 Vollständiger Pfad: {output_dir}\n\n"
                f"ℹ️ Kompatibel mit GLTF-Packer Standard\n"
                f"📖 Siehe {info_filename} für Installationsanweisungen"
            )
            
        except Exception as e:
            error_msg = f"Fehler beim GLTF Export: {str(e)}"
            print(f"❌ {error_msg}")
            self.status_var.set("❌ GLTF Export fehlgeschlagen")
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
        
        print(f"🔍 Auto-Find für Base: '{base_filename}' -> Basis-Name: '{base_name}'")
        
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
                print(f"✅ {tex_type}: '{best_match}' geladen (Score: {best_score})")
            elif best_match:
                print(f"⚠️ {tex_type}: '{best_match}' übersprungen (Score: {best_score} < 60)")
        
        self.status_var.set(f"✅ Auto-Find abgeschlossen: {found_count} Texturen gefunden")
    
    def generate_missing_maps(self):
        """Generiere fehlende Maps aus Base Color Textur"""
        try:
            # Prüfe ob Base Color verfügbar ist (erst in current_texture_images, dann in current_textures)
            base_image = None
            
            # 1. Zuerst prüfen ob bereits als PIL Image geladen
            if hasattr(self, 'current_texture_images') and "base_color" in self.current_texture_images and self.current_texture_images["base_color"] is not None:
                base_image = self.current_texture_images["base_color"]
                print("✅ Base Color aus Image-Cache geladen")
            
            # 2. Falls nicht im Cache, aus Pfad laden
            elif "base_color" in self.pbr_maker.current_textures and self.pbr_maker.current_textures["base_color"]:
                base_color_path = self.pbr_maker.current_textures["base_color"]
                if os.path.exists(base_color_path):
                    try:
                        base_image = Image.open(base_color_path)
                        print(f"✅ Base Color aus Datei geladen: {os.path.basename(base_color_path)}")
                    except Exception as e:
                        print(f"❌ Fehler beim Laden der Base Color: {e}")
            
            # 3. Wenn keine Base Color gefunden
            if base_image is None:
                messagebox.showwarning("Warnung", "Base Color Textur erforderlich!\nBitte zuerst eine Base Color Textur laden.")
                return
            self.status_var.set("✨ Generiere fehlende Maps...")
            
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
                result_text = f"✅ Generiert: {', '.join(maps_generated)}"
                self.status_var.set(result_text)
                messagebox.showinfo("Maps generiert", f"Erfolgreich generiert:\n• {chr(10).join(maps_generated)}")
                
                # 3D-Vorschau aktualisieren
                self.refresh_gltf_preview()
            else:
                self.status_var.set("ℹ️ Alle Maps bereits vorhanden")
                messagebox.showinfo("Maps generieren", "Alle Maps sind bereits vorhanden.")
                
        except Exception as e:
            print(f"❌ Fehler bei Map-Generierung: {e}")
            self.status_var.set("❌ Fehler bei Map-Generierung")
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
            print(f"❌ Fehler bei Normal Map Generierung: {e}")
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
            print(f"❌ Fehler bei Roughness Map Generierung: {e}")
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
            print(f"❌ Fehler bei Metallic Map Generierung: {e}")
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
            print(f"⚠️ Occlusion Map Generierung (Fallback): {e}")
            try:
                # Einfacher Fallback: Verdunkelte Version der Base Color
                gray = base_image.convert('L')
                gray_array = np.array(gray)
                
                # Leichte Verdunklung als einfache AO-Approximation
                occlusion_array = (gray_array * 0.8).astype(np.uint8)
                
                return Image.fromarray(occlusion_array, 'L').convert('RGB')
            except Exception as e2:
                print(f"❌ Fehler bei Occlusion Map Fallback: {e2}")
                return None
    
    def generate_emission_map_from_base(self, base_image):
        """Generiere Emission Map aus Base Color (helle Bereiche = emissiv)"""
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
            
            # Konvertiere zu RGB
            return Image.fromarray(emission_array, 'L').convert('RGB')
            
        except Exception as e:
            print(f"❌ Fehler bei Emission Map Generierung: {e}")
            return None
    
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
            print(f"❌ Fehler bei Alpha Map Generierung: {e}")
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
            print(f"❌ Fehler bei Height Map Generierung: {e}")
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
                
                print(f"✅ {texture_type} Map generiert und UI aktualisiert")
                
        except Exception as e:
            print(f"❌ Fehler beim Setzen der Textur {texture_type}: {e}")
        
    def apply_preset(self, event=None):
        """Wende Material-Preset an"""
        preset_name = self.preset_var.get()
        if preset_name and preset_name in self.pbr_maker.material_presets:
            preset = self.pbr_maker.material_presets[preset_name]
            
            # Aktualisiere Parameter
            param_mapping = {
                "normal_strength": "NormalStrength",
                "roughness_strength": "RoughnessStrength", 
                "occlusion_strength": "OcclusionStrength",
                "metallic_threshold": "MetallicThreshold",
                "emission_strength": "EmissionStrength",
                "alpha_strength": "AlphaStrength"
            }
            
            for param_key, config_key in param_mapping.items():
                if config_key in preset and param_key in self.param_vars:
                    self.param_vars[param_key].set(preset[config_key])
            
            self.status_var.set(f"✅ Preset '{preset_name}' angewendet")
    
    def save_material_package(self):
        """Speichere Material-Paket"""
        # Implementierung des GLTF-Exports
        self.status_var.set("💾 Material-Paket wird erstellt...")
        # TODO: Implementiere GLTF-Export
        
    def clear_all(self):
        """Lösche alle Texturen und setze Platzhalterbilder zurück"""
        for tex_type in self.pbr_maker.current_textures:
            self.pbr_maker.current_textures[tex_type] = None
            
            preview_label, file_label = self.texture_labels[tex_type]
            
            # Setze Platzhalterbild zurück
            self.set_placeholder_image(preview_label, tex_type)
            file_label.config(text="Keine Datei", foreground="gray")
        
        self.status_var.set("✅ Alle Texturen gelöscht")
    
    def change_thumbnail_size(self, event=None):
        """Ändere die Thumbnail-Größe und aktualisiere alle Vorschaubilder"""
        new_size = self.size_var.get()
        if new_size in self.size_options:
            self.current_size = new_size
            self.thumbnail_size = self.size_options[new_size]
            self.update_size_info()
            
            # Aktualisiere alle Vorschaubilder mit der neuen Größe
            for tex_type in self.texture_labels:
                preview_label, file_label = self.texture_labels[tex_type]
                
                # Lade Vorschaubilder neu wenn Texturen vorhanden sind
                if self.pbr_maker.current_textures[tex_type]:
                    self.update_texture_preview(tex_type, self.pbr_maker.current_textures[tex_type])
                else:
                    # Setze Platzhalterbild zurück
                    self.set_placeholder_image(preview_label, tex_type)
            
            self.status_var.set(f"✅ Vorschaubild-Größe auf {new_size} ({self.thumbnail_size[0]}x{self.thumbnail_size[1]}) geändert")
    
    def update_size_info(self):
        """Aktualisiere die Größen-Info Anzeige"""
        width, height = self.thumbnail_size
        self.size_info_var.set(f"Aktuelle Größe: {width}x{height} Pixel")
    
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
                self.status_var.set("✅ StandardPBR Preset geladen")
            else:
                self.status_var.set("❌ StandardPBR Preset nicht gefunden")
        except Exception as e:
            self.status_var.set(f"❌ Fehler beim Laden von StandardPBR: {e}")
    
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
                self.status_var.set("❌ Keine Bilddateien im gewählten Ordner gefunden")
                return
            
            # Verwende das erste gefundene Bild für alle Texturtypen
            primary_image = image_files[0]
            loaded_count = 0
            
            for tex_type in self.pbr_maker.current_textures:
                self.load_texture(tex_type, primary_image)
                loaded_count += 1
            
            self.status_var.set(f"✅ '{os.path.basename(primary_image)}' für alle {loaded_count} Texturtypen geladen + StandardPBR aktiviert")
            
        except Exception as e:
            self.status_var.set(f"❌ Fehler beim Laden des Bildersets: {e}")
    
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
                            
                        print(f"✅ Verwende '{placeholder_path}' für Größe '{size_name}'")
                else:
                    print(f"⚠️  Placeholder-Datei nicht gefunden: {placeholder_path}")
                    placeholders[size_key]["general"] = None
                    for tex_type in ["base_color", "normal", "roughness", "metallic", "occlusion", "emission", "alpha", "height"]:
                        placeholders[size_key][tex_type] = None
                        
            except Exception as e:
                print(f"❌ Fehler beim Laden des Placeholder-Bildes {placeholder_path}: {e}")
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
        preview_label.config(image="", text="📷 Ihre Vorlage", relief="sunken")
        preview_label.image = None
    
    def run(self):
        """Starte die GUI"""
        print("OpenSimulator PBR Material Maker - Tkinter Edition")
        print("Features:")
        print("- ✅ Echte Drag & Drop Unterstützung")
        print("- ✅ Bildvorschau mit PIL/Tkinter") 
        print("- ✅ Material-Presets")
        print("- ✅ Responsive GUI")
        print("- ✅ Bessere Benutzerfreundlichkeit")
        print("- ✅ StandardPBR Auto-Load")
        print("- ✅ Bilderset-Loader")
        print("- ✅ Ihre eigenen Platzhalterbilder aktiviert")
        
        # Beim Start automatisch StandardPBR laden
        self.root.after(100, self.load_standard_pbr)  # Nach 100ms laden
        
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = MaterialMakerGUI()
        app.run()
    except ImportError as e:
        if "tkinterdnd2" in str(e):
            print("❌ tkinterdnd2 nicht gefunden!")
            print("Installation: pip install tkinterdnd2")
        else:
            print(f"❌ Import-Fehler: {e}")
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()