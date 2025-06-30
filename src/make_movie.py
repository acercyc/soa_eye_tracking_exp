# Set up the figure with 300x300 pixels (3x3 inches at 100 dpi)
fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.axis("off")

# Create a white text object
text = ax.text(0.5, 0.5, '', fontsize=60, ha='center', va='center', color='white')

# Function to update the frame with white text, including 0 for 1 second
def update_with_zero(frame):
    text.set_text(str(5 - frame))
    return text,

# Now use 6 frames: 5, 4, 3, 2, 1, 0
countdown_animation_with_zero = animation.FuncAnimation(fig, update_with_zero, frames=6, interval=1000, blit=True)

# Save the updated animation
output_path_with_zero = "countdown_6_seconds_300x300.mp4"
countdown_animation_with_zero.save(output_path_with_zero, writer='ffmpeg', fps=1)

output_path_with_zero
