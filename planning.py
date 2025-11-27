# planning.py
import math


def dist_sq(p1, p2):
    """Distance au carré entre deux points (x, y)."""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def get_closest_cone_index(pos, cones):
    """Trouve l'index du cône le plus proche d'une position donnée."""
    if not cones:
        return None

    min_dist = float('inf')
    best_idx = -1

    for i, cone in enumerate(cones):
        d = dist_sq(pos, (cone['x'], cone['y']))
        if d < min_dist:
            min_dist = d
            best_idx = i

    return best_idx


def compute_center_line(cones):
    """
    Génère une liste de points (x, y) passant au milieu des cônes bleus et jaunes.
    """
    # 1. Séparer les cônes par couleur
    blues = [c for c in cones if c['tag'] == 'blue']
    yellows = [c for c in cones if c['tag'] == 'yellow']

    # Trouver la position de départ
    start_pos = (0.0, 0.0)
    for c in cones:
        if c['tag'] == 'car_start':
            start_pos = (c['x'], c['y'])
            break

    path = [start_pos]
    current_pos = start_pos

    # Paramètre de sécurité : si les cônes sont trop écartés (fin de piste), on arrête
    MAX_TRACK_WIDTH_SQ = 15.0 ** 2  # 15 mètres max de large

    # 2. Algorithme glouton (Zipper method)
    # On continue tant qu'il reste des cônes des deux couleurs
    while len(blues) > 0 and len(yellows) > 0:

        # Trouver le bleu le plus proche de la voiture (ou du dernier point)
        b_idx = get_closest_cone_index(current_pos, blues)
        b_cone = blues[b_idx]

        # Trouver le jaune le plus proche de la voiture
        y_idx = get_closest_cone_index(current_pos, yellows)
        y_cone = yellows[y_idx]

        # Vérification de sécurité : si les deux cônes trouvés sont trop loin l'un de l'autre,
        # c'est qu'on essaie probablement de relier des parties non connectées de la piste.
        if dist_sq((b_cone['x'], b_cone['y']), (y_cone['x'], y_cone['y'])) > MAX_TRACK_WIDTH_SQ:
            break

        # 3. Calculer le point milieu (Midpoint)
        mid_x = (b_cone['x'] + y_cone['x']) / 2
        mid_y = (b_cone['y'] + y_cone['y']) / 2

        next_point = (mid_x, mid_y)
        path.append(next_point)
        current_pos = next_point

        # 4. Retirer les cônes utilisés pour avancer aux suivants
        # On retire celui qu'on vient d'utiliser pour ne pas tourner en rond
        blues.pop(b_idx)
        yellows.pop(y_idx)

    # 5. Lissage simple de la trajectoire (Moyenne mobile)
    # Cela rend le mouvement de la voiture plus fluide
    smoothed_path = []
    if len(path) > 2:
        smoothed_path.append(path[0])  # Garder le point de départ
        for i in range(1, len(path) - 1):
            prev_p = path[i - 1]
            curr_p = path[i]
            next_p = path[i + 1]

            # Moyenne des voisins
            avg_x = (prev_p[0] + curr_p[0] + next_p[0]) / 3
            avg_y = (prev_p[1] + curr_p[1] + next_p[1]) / 3
            smoothed_path.append((avg_x, avg_y))
        smoothed_path.append(path[-1])  # Garder le dernier point
        return smoothed_path

    return path