import time
import pygame
import sys
from pythonosc import udp_client
import random


class Slider:
    def __init__(self, x, y, width, height, min_val=10, max_val=117, initial_val=63):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.dragging = False

        self.bg_color = (100, 100, 100)
        self.handle_color = (200, 200, 200)
        self.handle_hover_color = (255, 255, 255)

        self.handle_width = 20
        self.handle_height = height

    def handle_rect(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + ratio * (self.rect.width - self.handle_width)
        return pygame.Rect(handle_x, self.rect.y, self.handle_width, self.handle_height)

    def handle_event(self, event):
        handle_rect = self.handle_rect()
        if event.type == pygame.MOUSEBUTTONDOWN and handle_rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            relative_x = event.pos[0] - self.rect.x
            ratio = max(
                0,
                min(
                    1,
                    (relative_x - self.handle_width / 2)
                    / (self.rect.width - self.handle_width),
                ),
            )
            new_val = int(self.min_val + ratio * (self.max_val - self.min_val))
            if new_val != self.val:
                self.val = new_val
                return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, (50, 50, 50), self.rect, 2)
        handle_rect = self.handle_rect()
        mouse_pos = pygame.mouse.get_pos()
        color = (
            self.handle_hover_color
            if handle_rect.collidepoint(mouse_pos) or self.dragging
            else self.handle_color
        )
        pygame.draw.rect(screen, color, handle_rect)
        pygame.draw.rect(screen, (50, 50, 50), handle_rect, 2)


class Button:
    def __init__(
        self,
        rect,
        label,
        font,
        on_click,
        bg=(70, 70, 70),
        hover=(90, 90, 90),
        text=(255, 255, 255),
        border=(30, 30, 30),
    ):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font
        self.on_click = on_click
        self.bg = bg
        self.hover = hover
        self.text = text
        self.border = border

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()
                return True
        return False

    def draw(self, screen):
        is_hover = self.rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(
            screen,
            self.hover if is_hover else self.bg,
            self.rect,
            border_radius=8,
        )
        pygame.draw.rect(screen, self.border, self.rect, 2, border_radius=8)
        surf = self.font.render(self.label, True, self.text)
        screen.blit(surf, surf.get_rect(center=self.rect.center))


class ToggleButton(Button):
    def __init__(self, rect, label, font, code, on_toggle, **kwargs):
        super().__init__(rect, label, font, None, **kwargs)
        self.code = code
        self.active = False
        self.on_toggle = on_toggle

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
                self.on_toggle(self.code, self.active)
                return True
        return False

    def draw(self, screen):
        is_hover = self.rect.collidepoint(pygame.mouse.get_pos())

        if self.active:
            base_color = (100, 200, 120)
        else:
            base_color = self.bg if not is_hover else self.hover

        pygame.draw.rect(screen, base_color, self.rect, border_radius=8)
        pygame.draw.rect(screen, self.border, self.rect, 2, border_radius=8)

        surf = self.font.render(self.label, True, self.text)
        screen.blit(surf, surf.get_rect(center=self.rect.center))


class OSCSliderApp:
    def __init__(self):
        pygame.init()

        self.width = 800
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("OSC Slider Control")

        self.bg_color = (40, 40, 40)
        self.text_color = (255, 255, 255)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 30)

        slider_width, slider_height = 400, 40
        slider_x = (self.width - slider_width) // 2

        # Sliders
        self.slider1 = Slider(slider_x, 200, slider_width, slider_height)
        self.slider2 = Slider(slider_x, 300, slider_width, slider_height)
        self.slider3 = Slider(slider_x, 600, slider_width, slider_height, 10, 100, 55)

        self.interval = (-8, 8)
        self.osc_client = udp_client.SimpleUDPClient("100.101.30.29", 9001)
        self.clock = pygame.time.Clock()
        self.running = True
        self.slider_interval = 0.1
        self.inactivity_interval = 5.0
        self.last_slider_change_time = time.time()
        self.inactivity_message_sent = False

        # ---- Instrument Toggle Buttons ----
        instr_codes = [0, 26, 32, 56, 128]
        btn_w, btn_h = 120, 40
        gap = 16
        total_w = len(instr_codes) * btn_w + (len(instr_codes) - 1) * gap
        start_x_instr = (self.width - total_w) // 2
        instr_y = 70

        self.active_instruments = set()
        self.instrument_buttons = []
        for i, code in enumerate(instr_codes):
            x = start_x_instr + i * (btn_w + gap)
            self.instrument_buttons.append(
                ToggleButton(
                    (x, instr_y, btn_w, btn_h),
                    str(code),
                    self.small_font,
                    code,
                    self.on_toggle_instrument
                )
            )

        # ---- Clear Pitch Button ----
        clear_w, clear_h = 180, 42
        clear_x = (self.width - clear_w) // 2
        clear_y = instr_y + btn_h + 20

        self.clear_pitch_btn = Button(
            (clear_x, clear_y, clear_w, clear_h),
            "Clear Pitch",
            self.small_font,
            lambda: self.send_osc_message(20, 120, -1, -1)
        )

        # ---- Tempo Buttons ----
        gap, btn_w, btn_h = 18, 180, 44
        total_w_row = 3 * btn_w + 2 * gap
        start_x_row = (self.width - total_w_row) // 2
        row1_y = 400
        row2_y = row1_y + btn_h + 16
        row3_y = row2_y + btn_h + 16
        total_w_row3 = 2 * btn_w + gap
        start_x_row3 = (self.width - total_w_row3) // 2

        self.buttons = [
            Button((start_x_row + 0 * (btn_w + gap), row1_y, btn_w, btn_h), "Free", self.small_font, lambda: self.send_tempo(0)),
            Button((start_x_row + 1 * (btn_w + gap), row1_y, btn_w, btn_h), "Short", self.small_font, lambda: self.send_tempo(1)),
            Button((start_x_row + 2 * (btn_w + gap), row1_y, btn_w, btn_h), "Long", self.small_font, lambda: self.send_tempo(2)),

            Button((start_x_row + 0 * (btn_w + gap), row2_y, btn_w, btn_h), "Fast", self.small_font, lambda: self.send_tempo(3)),
            Button((start_x_row + 1 * (btn_w + gap), row2_y, btn_w, btn_h), "Med", self.small_font, lambda: self.send_tempo(4)),
            Button((start_x_row + 2 * (btn_w + gap), row2_y, btn_w, btn_h), "Slow", self.small_font, lambda: self.send_tempo(5)),

            Button((start_x_row3 + 0 * (btn_w + gap), row3_y, btn_w, btn_h), "Runs", self.small_font, lambda: self.send_tempo_with_param(6, self.slider3.val)),
            Button((start_x_row3 + 1 * (btn_w + gap), row3_y, btn_w, btn_h), "Chords", self.small_font, lambda: self.send_tempo_with_param(7, self.slider3.val)),
        ]

    # ---- Instrument OSC ----
    def on_toggle_instrument(self, code, is_active):
        if is_active:
            self.active_instruments.add(code)
        else:
            self.active_instruments.discard(code)

        arr = sorted(self.active_instruments)
        self.osc_client.send_message("/setActiveInstruments", arr)
        print(f"Sent OSC: /setActiveInstruments {arr}")

    # ---- Range OSC ----
    def send_osc_message(self, a=None, b=None, c=None, d=None):
        if any(v is None for v in (a, b, c, d)):
            v1, v2 = self.slider1.val, self.slider2.val
            a, b = v1 + self.interval[0], v1 + self.interval[1]
            c, d = v2 + self.interval[0], v2 + self.interval[1]
        self.osc_client.send_message("/setOutputRange", [a, b, c, d])
        print(f"Sent OSC: /setOutputRange {a} {b} {c} {d}")

    # ---- Tempo OSC ----
    def send_tempo(self, value):
        self.osc_client.send_message("/setTempo", value)
        print(f"Sent OSC: /setTempo {value}")

    def send_tempo_with_param(self, value, param):
        self.osc_client.send_message("/setTempo", [value, param])
        print(f"Sent OSC: /setTempo {value} {param}")

    # ---- Events ----
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            changed = any([
                self.slider1.handle_event(event),
                self.slider2.handle_event(event),
            ])
            self.slider3.handle_event(event)

            if changed:
                now = time.time()
                if now - self.last_slider_change_time >= self.slider_interval:
                    self.last_slider_change_time = now
                    self.inactivity_message_sent = False
                    self.send_osc_message()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in self.buttons:
                    btn.handle_event(event)
                for btn in self.instrument_buttons:
                    btn.handle_event(event)

                self.clear_pitch_btn.handle_event(event)

    # ---- Inactivity ----
    def handle_inactivity(self):
        if time.time() - self.last_slider_change_time >= self.inactivity_interval and not self.inactivity_message_sent:
            self.send_osc_message(0, 127, 0, 127)
            self.inactivity_message_sent = True

    # ---- Draw ----
    def draw(self):
        self.screen.fill(self.bg_color)
        title = self.font.render("OSC Slider Control", True, self.text_color)
        self.screen.blit(title, title.get_rect(center=(self.width // 2, 40)))

        for btn in self.instrument_buttons:
            btn.draw(self.screen)

        self.clear_pitch_btn.draw(self.screen)

        labels = [
            ("Right Pitch", self.slider1),
            ("Left Pitch", self.slider2),
            ("Beat Length", self.slider3),
        ]
        for name, slider in labels:
            lbl = self.small_font.render(f"{name}: {slider.val}", True, self.text_color)
            self.screen.blit(lbl, (slider.rect.x, slider.rect.y - 28))
            slider.draw(self.screen)

        for btn in self.buttons:
            btn.draw(self.screen)

        pygame.display.flip()

    # ---- Main Loop ----
    def run(self):
        while self.running:
            self.handle_events()
            self.handle_inactivity()
            self.draw()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = OSCSliderApp()
    app.run()
