# python 3.6.5
class Vehicle:

    def move(self):
        print('move!')

    def transport(self, thing, to):
        print('transport %s to %s' % (thing, to))


class Car(Vehicle):
    width = 0
    height = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def speed_up(self, to):
        print('speed up to %dkm/h' % (to))

    def slow_down(self, to):
        print('slow down to %dkm/h' % (to))


car = Car(5, 5)
car.move()
car.transport('book', 'sspku')
car.speed_up(180)
car.slow_down(60)
