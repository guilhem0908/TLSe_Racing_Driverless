import pygame
import math
from typing import List, Tuple

from track_utils import get_start_pos
from simulation.camera import Camera
from simulation.vision import VisionConfig, DEFAULT_VISION, vision_cone_polygon_world, point_in_vision_cone

# --- Paramètres de la voiture ---
CAR_SPEED = 5.0  # Vitesse constante (m/s)
TURN_SPEED = 5.0  # Vitesse de braquage (facteur de lissage)
CAR_SIZE_M = 1.5  # Taille visuelle

# Couleurs
BG_COLOR = (30, 30, 30)
COLOR_CAR = (255, 0, 255)
COLOR_VISION = (160, 160, 160, 60)
COLOR_TARGET = (255, 50, 50)  # Point rouge que la voiture vise


def dist_sq(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def get_closest(pos, cones):
    """Trouve le cône le plus proche dans une liste."""
    if not cones:
        return None
    best_c = None
    min_d = float('inf')
    for c in cones:
        d = dist_sq(pos, (c['x'], c['y']))
        if d < min_d:
            min_d = d
            best_c = c
    return best_c


def run_realtime(cones, world_bounds):
    pygame.init()
    screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
    pygame.display.set_caption("Pilotage Temps Réel - Vision Cone")
    clock = pygame.time.Clock()

    # 1. Initialisation de l'état de la voiture (Physique)
    start_x, start_y = get_start_pos(cones)

    # On essaie de deviner l'angle de départ (regarder vers le 1er cone bleu)
    # Sinon 0 par défaut
    car_x, car_y = start_x, start_y
    car_heading_deg = 0.0

    # Caméra
    camera = Camera(world_bounds, screen.get_size())

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time en secondes
        screen_size = screen.get_size()

        # --- Gestion des événements (Zoom/Pan/Quit) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                camera = Camera(world_bounds, screen.get_size())
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.change_zoom(1.1, pygame.mouse.get_pos(), screen_size)
                elif event.y < 0:
                    camera.change_zoom(1 / 1.1, pygame.mouse.get_pos(), screen_size)

        # --- 2. PERCEPTION (Vision) ---
        # On ne garde que les cônes qui sont DANS le cône de vision
        visible_cones = []
        for c in cones:
            # On vérifie si le point (c['x'], c['y']) est dans le triangle de vision
            if point_in_vision_cone((c['x'], c['y']), (car_x, car_y), car_heading_deg, DEFAULT_VISION):
                visible_cones.append(c)

        # Séparer par couleur
        blues = [c for c in visible_cones if c['tag'] == 'blue']
        yellows = [c for c in visible_cones if c['tag'] == 'yellow']

        # --- 3. PLANIFICATION (Décision) ---
        target_x, target_y = None, None

        # Stratégie : Viser le milieu entre le bleu le plus proche et le jaune le plus proche
        closest_blue = get_closest((car_x, car_y), blues)
        closest_yellow = get_closest((car_x, car_y), yellows)

        if closest_blue and closest_yellow:
            target_x = (closest_blue['x'] + closest_yellow['x']) / 2
            target_y = (closest_blue['y'] + closest_yellow['y']) / 2
        elif closest_blue:
            # Si on ne voit que du bleu, on s'écarte un peu (stratégie de survie)
            target_x = closest_blue['x'] + 2.0  # Décalage arbitraire
            target_y = closest_blue['y'] + 2.0
        elif closest_yellow:
            target_x = closest_yellow['x'] - 2.0
            target_y = closest_yellow['y'] - 2.0

        # --- 4. CONTROLE (Action) ---
        if target_x is not None:
            # Calcul de l'angle vers la cible
            dx = target_x - car_x
            dy = target_y - car_y
            target_angle = math.degrees(math.atan2(dy, dx))

            # Lissage de la direction (Direction assistée)
            # On tourne doucement vers l'angle cible pour ne pas faire d'à-coups
            diff = (target_angle - car_heading_deg + 180) % 360 - 180
            car_heading_deg += diff * TURN_SPEED * dt

        # --- 5. PHYSIQUE (Mouvement) ---
        # Avancer dans la direction actuelle
        rad = math.radians(car_heading_deg)
        car_x += math.cos(rad) * CAR_SPEED * dt
        car_y += math.sin(rad) * CAR_SPEED * dt

        # --- 6. RENDU (Dessin) ---
        screen.fill(BG_COLOR)

        # Dessiner le cône de vision
        poly_world = vision_cone_polygon_world((car_x, car_y), car_heading_deg, DEFAULT_VISION)
        poly_screen = [camera.world_to_screen(px, py, screen_size) for px, py in poly_world]

        # Surface transparente pour le cône
        overlay = pygame.Surface(screen_size, pygame.SRCALPHA)
        pygame.draw.polygon(overlay, COLOR_VISION, poly_screen)
        screen.blit(overlay, (0, 0))

        # Dessiner TOUS les cônes (les visibles en surbrillance)
        for c in cones:
            sx, sy = camera.world_to_screen(c['x'], c['y'], screen_size)
            color = (50, 100, 255) if c['tag'] == 'blue' else (255, 255, 0)
            if c['tag'] == 'big_orange': color = (255, 100, 0)

            # Si le cône est dans la liste 'visible_cones', on le dessine en gros/brillant
            radius = 6 if c in visible_cones else 3
            if c in visible_cones:
                pygame.draw.circle(screen, (255, 255, 255), (sx, sy), radius + 2)  # Contour blanc
            pygame.draw.circle(screen, color, (sx, sy), radius)

        # Dessiner la voiture
        cx_screen, cy_screen = camera.world_to_screen(car_x, car_y, screen_size)
        car_rect = pygame.Surface((20, 10), pygame.SRCALPHA)
        car_rect.fill(COLOR_CAR)
        # Rotation visuelle (attention Pygame tourne dans le sens trigo inverse parfois selon l'axe Y)
        car_rotated = pygame.transform.rotate(car_rect, car_heading_deg)
        rect = car_rotated.get_rect(center=(cx_screen, cy_screen))
        screen.blit(car_rotated, rect)

        # Dessiner la cible (Debug)
        if target_x is not None:
            tx_s, ty_s = camera.world_to_screen(target_x, target_y, screen_size)
            pygame.draw.line(screen, COLOR_TARGET, (cx_screen, cy_screen), (tx_s, ty_s), 2)
            pygame.draw.circle(screen, COLOR_TARGET, (tx_s, ty_s), 5)

        pygame.display.flip()

    pygame.quit()