import pygame as pg
import numpy as np


WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 900
FPS = 145

DENSITY = 1
PARTICLE_SIZE = 40
PI = np.pi
NUM_PARTICLES = 5

GRAVITY = np.array((0, 0.15), np.float32)

SHOW_TRAILS = True

quadtreeCapacity = 1
entropy = 0.1

acceptableVectorDataTypes = (np.ndarray, list, tuple, pg.Vector2)

pg.init()

font = pg.font.Font(size=30)
screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pg.RESIZABLE, pg.FULLSCREEN)
clock = pg.time.Clock()


class Particle:
    def __init__(self, size=PARTICLE_SIZE, pos=None, vel=None) -> None:
        self.collided = None
        self.collisionTime = 0
        self.density = DENSITY
        self.size = size
        self.radius = size / 2
        self.mass = self.density * PI * (self.radius ** 2)

        self.surfWhite = pg.transform.scale(pg.image.load('pythonProject\\particleWhite.png'), (size, size))
        self.surfRed = pg.transform.scale(pg.image.load('pythonProject\\particleRed.png'), (size, size))

        if pos: self.pos = np.array((pos[0] - self.radius, pos[1] - self.radius), np.float64)
        else: self.pos = np.array((np.random.random() * (WINDOW_WIDTH - size + 1), np.random.random() * (WINDOW_HEIGHT - size + 1)),
                                  np.float64) if type(pos) not in acceptableVectorDataTypes else pos

        if vel: self.vel = vel
        else: self.vel = np.array((np.random.choice((-1, 1)) * np.random.random() * 0.2, np.random.choice((-1, 1)) * np.random.random() * 0.2),
                                  np.float64) if type(vel) not in acceptableVectorDataTypes else vel

        self.offset = np.array((self.radius, self.radius), np.int8)
        self.centerPos = self.pos + self.offset

        self.trails = [tuple(self.pos)]


def renderUI(display):
    display.blit(font.render(f'FPS: {round(clock.get_fps(), 2)}', True, (255, 255, 255)), (0, 5))
    display.blit(font.render(f'No. of Particles: {len(particles)}', True, (255, 255, 255)), (0, 30))


def endSimulation() -> None: pg.quit(), exit()


def boxCheck(particle: Particle):
    x, y = particle.pos
    size = particle.size

    if x >= WINDOW_WIDTH - size:  # Right border
        particle.vel[0] *= -1
        particle.pos[0] = WINDOW_WIDTH - size

    if x <= 0:  # Left border
        particle.vel[0] *= -1
        particle.pos[0] = 0

    if y <= 0:  # Top border
        particle.vel[1] *= -1
        particle.pos[1] = 0

    if y >= WINDOW_HEIGHT - size:  # Bottom border
        particle.vel[1] *= -1
        particle.pos[1] = WINDOW_HEIGHT - size


def resolveCollision(p1: Particle, p2: Particle):
    p1.collided = p2.collided = True
    p1.collisionTime = p2.collisionTime = pg.time.get_ticks()

    collisionNormal = p1.pos - p2.pos
    distance = np.linalg.norm(collisionNormal)  # magnitude of vec

    if not distance: return  # to avoid div by zero
    collisionNormalUnit = collisionNormal / distance

    relVel = p1.vel - p2.vel
    velAlongNormal = np.dot(relVel, collisionNormalUnit)

    if velAlongNormal > 0:  # both are in the same-ish direction
        return  # cuz they are moving away from each other

    e = 0.99  # dampening the collisions to avoid inf energy in the sys
    j = -(1 + e) * velAlongNormal
    j /= np.reciprocal(p1.mass) + np.reciprocal(p2.mass)

    impulse = j * collisionNormalUnit
    p1.vel += impulse / p1.mass
    p2.vel -= impulse / p2.mass


def drawParticles() -> None:
    for particle in particles:
        if pg.time.get_ticks() - particle.collisionTime > 125: particle.collided = False

        if particle.collided:
            screen.blit(particle.surfRed, particle.pos)
        else:
            screen.blit(particle.surfWhite, particle.pos)

        if SHOW_TRAILS: [pg.draw.circle(screen, (150, 150, 150), point, particle.size / 12) for point in particle.trails]


def checkCollision() -> None:
    for particle in particles:
        particle.pos += particle.vel
        if particle.pos[1] < WINDOW_HEIGHT - particle.size: particle.vel += GRAVITY

        boxCheck(particle)
        particle.trails.append(particle.pos + particle.offset)
        particle.trails = particle.trails[-15:]
        # particle.vel = (particle.vel + gravity) if particle.pos[1] not in (WINDOW_HEIGHT-particle.size, ) else particle.vel

    for i, p1 in enumerate(particles):
        for p2 in particles[i + 1:]:
            if np.linalg.norm(p1.pos - p2.pos) < (p1.radius + p2.radius):
                resolveCollision(p1, p2)


def computeME():
    tKE = tPE = 0
    for particle in particles:
        v = np.linalg.norm(particle.vel)

        KE = particle.mass * (v ** 2) * 0.5
        PE = particle.mass * GRAVITY[1] * (WINDOW_HEIGHT - particle.centerPos[1])

        tKE += KE
        tPE += PE

    # print(f'tME: {tPE + tKE:.2f}')


particles = [Particle() for _ in range(NUM_PARTICLES)]

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT: endSimulation()

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE: endSimulation()

            if event.key == pg.K_EQUALS:
                particles.append(Particle())
                print(f'No. of particles: {len(particles)}')

            if event.key == pg.K_MINUS:
                print(f'No. of particles: {len(particles)}')
                particles.pop(0) if len(particles) != 0 else print('No particles left to remove.')

            if event.key == pg.K_0:
                FPS = 145
                print(f'FPS reset to 120.')

            if event.key == pg.K_LEFTBRACKET:
                FPS -= 10 if FPS > 5 else 0
                print(f'FPS: {FPS}')

            if event.key == pg.K_RIGHTBRACKET:
                FPS += 10
                print(f'FPS: {FPS}')

            if event.key == pg.K_BACKSLASH:
                FPS = 0 if FPS else 145

            if event.key == pg.K_SPACE:
                SHOW_TRAILS = not SHOW_TRAILS

        if event.type == pg.MOUSEBUTTONDOWN:
            particles.append(Particle(pos=pg.mouse.get_pos()))
            print(f'No. of particles: {len(particles)}')

    screen.fill((0, 0, 0))
    checkCollision()
    drawParticles()
    computeME()
    renderUI(screen)

    pg.display.update()
    clock.tick(FPS)
