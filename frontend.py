import imageio
import pygame
import sys
import matplotlib.pyplot as plt
import numpy as np
from letter_recognition import predict_letter

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
frame_thickness = 80
button_height = 50
screen = pygame.display.set_mode((width, height + button_height))
pygame.display.set_caption("AI will guess your letter")

# Set up colors
black = (0, 0, 0)
grey = (220, 220, 220)
dark_grey = (200, 200, 200)
white = (255, 255, 255)

# Set up drawing variables
drawing = False
last_pos = None
radius = 40

# Set up button variables
button1_width, button2_width = 100, 100
yes_button_rect: pygame.Rect = None
no_button_rect: pygame.Rect = None

# Adjust the starting position of buttons for better centering
button1_rect = pygame.Rect((width - button1_width) // 4, height + 10, button1_width, button_height)
button2_rect = pygame.Rect((3 * width - button2_width) // 4, height + 10, button2_width, button_height)
button_font = pygame.font.Font(None, 36)

# Set background color
screen.fill(white)

# Set up popup variables
popup_width, popup_height = 300, 150
popup_rect = pygame.Rect((width - popup_width) // 2, (height - popup_height) // 2, popup_width, popup_height)
popup_font = pygame.font.Font(None, 24)

# Variable to control the visibility of the popup
show_popup = None


def draw(last_pos, current_pos):
    if (last_pos is None):
        return
    pygame.draw.line(screen, black, last_pos, current_pos, radius)


def save_image():
    # TODO Kevin: Lav mappe, gør så den gemmer drawing_1, drawing_2.... så den ikke overskriver hvergang
    pygame.image.save(screen.subsurface(
        (frame_thickness, frame_thickness, width - 2 * frame_thickness, height - 2 * frame_thickness)), "drawing.png")


# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button1_rect.collidepoint(event.pos):
                print("This is what the AI guessed: ")
                # call function to convert drawing to 28x28 png
                save_image()
                result = predict_letter("drawing.png", show_converted_letter=True)
                # convert the RGB values to greyscale
                print(result)

                # Show the popup
                show_popup = True

            elif button2_rect.collidepoint(event.pos):
                print("Canvas has been cleared")
                screen.fill(white)
                # Draw frame after clearing to keep the border
                pygame.draw.rect(screen, grey, (0, 0, width, frame_thickness))
                pygame.draw.rect(screen, grey, (0, height - frame_thickness, width, frame_thickness))
                pygame.draw.rect(screen, grey, (0, 0, frame_thickness, height))
                pygame.draw.rect(screen, grey, (width - frame_thickness, 0, frame_thickness, height))

            # Check if "Yes" button is clicked
            elif yes_button_rect is not None and yes_button_rect.collidepoint(event.pos):
                print("Yes Button Clicked")
                show_popup = False  # Close popup

            # Check if "No" button is clicked
            elif no_button_rect is not None and no_button_rect.collidepoint(event.pos):
                print("No Button Clicked")
                show_popup = False  # Close popup

            else:
                drawing = True
                last_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                draw(last_pos, event.pos)
                last_pos = event.pos

    # Draw frame
    pygame.draw.rect(screen, grey, (0, 0, width, frame_thickness))  # Top
    pygame.draw.rect(screen, grey, (0, height - frame_thickness, width, frame_thickness))  # Bottom
    pygame.draw.rect(screen, grey, (0, 0, frame_thickness, height))  # Left
    pygame.draw.rect(screen, grey, (width - frame_thickness, 0, frame_thickness, height))  # Right

    # Draw buttons
    pygame.draw.rect(screen, dark_grey, button1_rect)
    pygame.draw.rect(screen, dark_grey, button2_rect)

    # Draw text on buttons
    button1_text = button_font.render("Confirm", True, black)
    button2_text = button_font.render("Clear", True, black)
    screen.blit(button1_text, (
        button1_rect.centerx - button1_text.get_width() // 2, button1_rect.centery - button1_text.get_height() // 2))
    screen.blit(button2_text, (
        button2_rect.centerx - button2_text.get_width() // 2, button2_rect.centery - button2_text.get_height() // 2))

    if show_popup:
        # Create the popup
        pygame.draw.rect(screen, grey, popup_rect)
        pygame.draw.rect(screen, dark_grey, popup_rect, 5)  # Draw a border around the popup

        # Draw text on the popup
        popup_text = popup_font.render("Did the AI guess correct?", True, black)
        screen.blit(popup_text, (popup_rect.centerx - popup_text.get_width() // 2,
                                 popup_rect.centery - popup_text.get_height() // 2 - 20))

        # Draw two buttons on the popup
        yes_button_rect = pygame.Rect(popup_rect.centerx - 60, popup_rect.centery + 20, 50, 30)
        no_button_rect = pygame.Rect(popup_rect.centerx + 10, popup_rect.centery + 20, 50, 30)

        pygame.draw.rect(screen, dark_grey, yes_button_rect)
        pygame.draw.rect(screen, dark_grey, no_button_rect)

        yes_button_text = popup_font.render("Yes", True, black)
        no_button_text = popup_font.render("No", True, black)

        screen.blit(yes_button_text,
                    (yes_button_rect.centerx - yes_button_text.get_width() // 2,
                     yes_button_rect.centery - yes_button_text.get_height() // 2))
        screen.blit(no_button_text,
                    (no_button_rect.centerx - no_button_text.get_width() // 2,
                     no_button_rect.centery - no_button_text.get_height() // 2))

    elif show_popup is False:
        screen = pygame.display.set_mode((width, height + button_height))
        pygame.display.set_caption("AI will guess your letter")

        # Set up colors
        black = (0, 0, 0)
        grey = (220, 220, 220)
        dark_grey = (200, 200, 200)
        white = (255, 255, 255)

        # Set up drawing variables
        drawing = False
        last_pos = None
        radius = 40

        # Set up button variables
        button1_width, button2_width = 100, 100
        yes_button_rect: pygame.Rect = None
        no_button_rect: pygame.Rect = None

        # Adjust the starting position of buttons for better centering
        button1_rect = pygame.Rect((width - button1_width) // 4, height + 10, button1_width, button_height)
        button2_rect = pygame.Rect((3 * width - button2_width) // 4, height + 10, button2_width, button_height)
        button_font = pygame.font.Font(None, 36)

        # Set background color
        screen.fill(white)

        show_popup = None

    # Update display
    pygame.display.flip()

# End the program
pygame.quit()
