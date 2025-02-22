from PIL import Image, ImageDraw, ImageFont
import os

def generate_dashboard_preview():
    # Create a new image
    width, height = 1200, 800
    img = Image.new('RGB', (width, height), color='#1E293B')
    draw = ImageDraw.Draw(img)
    
    # Add some mock UI elements
    draw.rectangle([50, 50, width-50, height-50], outline='#3B82F6', width=2)
    draw.text((width/2, height/2), 'Seismic Dashboard Preview', fill='#FFFFFF', anchor='mm')
    
    # Save the image
    img_path = os.path.join('images', 'dashboard-preview.jpg')
    img.save(img_path, 'JPEG', quality=95)

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)
    generate_dashboard_preview()
