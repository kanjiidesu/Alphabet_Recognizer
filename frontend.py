import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
frame_thickness = 80
button_height = 50
screen = pygame.display.set_mode((width, height + button_height))
pygame.display.set_caption("Simple Draw Program")

# Set up colors
black = (0, 0, 0)
grey = (220, 220, 220)
dark_grey = (200, 200, 200)
white = (255, 255, 255)

# Set up drawing variables
drawing = False
last_pos = None
radius = 7

# Set up button variables
button1_width, button2_width = 100, 100
# Adjust the starting position of buttons for better centering
button1_rect = pygame.Rect((width - button1_width) // 4, height + 10, button1_width, button_height)
button2_rect = pygame.Rect((3 * width - button2_width) // 4, height + 10, button2_width, button_height)
button_font = pygame.font.Font(None, 36)

# Set background color
screen.fill(white)


def draw(last_pos, current_pos):
    if (last_pos is None):
        return
    pygame.draw.line(screen, black, last_pos, current_pos, radius)


# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button1_rect.collidepoint(event.pos):
                # instead of below clear, print what AI guesses (what letter is it?)
                print("Button 1 Clicked")
                # call function to convert drawing to 28x28 png
                # call AI
                # Add your button 1 action here

            elif button2_rect.collidepoint(event.pos):
                print("Canvas has been cleared")
                screen.fill(white)
                # Draw frame after clearing to keep the border
                pygame.draw.rect(screen, grey, (0, 0, width, frame_thickness))
                pygame.draw.rect(screen, grey, (0, height - frame_thickness, width, frame_thickness))
                pygame.draw.rect(screen, grey, (0, 0, frame_thickness, height))
                pygame.draw.rect(screen, grey, (width - frame_thickness, 0, frame_thickness, height))

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
    screen.blit(button1_text, (button1_rect.centerx - button1_text.get_width() // 2, button1_rect.centery - button1_text.get_height() // 2))
    screen.blit(button2_text, (button2_rect.centerx - button2_text.get_width() // 2, button2_rect.centery - button2_text.get_height() // 2))


    # Update display
    pygame.display.flip()

# End the program
pygame.quit()
