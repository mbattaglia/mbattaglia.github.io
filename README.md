# My Personal Blog

A simple, clean personal blog built with Jekyll and hosted on GitHub Pages using the Minima theme.

## 🌐 Live Site

Visit the blog at: [https://yourusername.github.io](https://yourusername.github.io)

## 🛠️ Built With

- **[Jekyll](https://jekyllrb.com/)** - Static site generator
- **[Minima Theme](https://github.com/jekyll/minima)** - Clean, minimal Jekyll theme
- **[GitHub Pages](https://pages.github.com/)** - Free hosting and deployment
- **[Markdown](https://daringfireball.net/projects/markdown/)** - Content writing format

## 📁 Project Structure

```
├── _config.yml           # Site configuration
├── _posts/               # Blog posts
├── assets/               # CSS, JS, images
├── about.md              # About page
├── talks.md              # Conference talks page
├── resources.md          # Resources and projects page
├── index.md              # Homepage
├── Gemfile               # Ruby dependencies
└── .gitignore           # Git ignore rules
```

## 🚀 Local Development

### Prerequisites

- Ruby (version 2.7+)
- Bundler
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yourusername.github.io.git
   cd yourusername.github.io
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Run the site locally**
   ```bash
   bundle exec jekyll serve
   ```

4. **View in browser**
   Open [http://localhost:4000](http://localhost:4000)

### Development Commands

```bash
# Serve site with live reload
bundle exec jekyll serve --livereload

# Serve site with drafts
bundle exec jekyll serve --drafts

# Build site for production
bundle exec jekyll build

# Update dependencies
bundle update
```

## ✍️ Writing Posts

### Creating a New Post

1. Create a new file in `_posts/` with the format: `YYYY-MM-DD-title.md`
2. Add front matter:
   ```yaml
   ---
   layout: post
   title: "Your Post Title"
   date: 2025-05-29 10:00:00 +0100
   categories: jekyll update
   tags: [tag1, tag2]
   excerpt: "Brief description"
   ---
   ```
3. Write your content in Markdown
4. Test locally, then commit and push to publish

### Post Guidelines

- Use descriptive titles
- Add relevant categories and tags
- Include an excerpt for better SEO
- Use proper Markdown formatting
- Test locally before publishing

## 🎨 Customization

### Site Configuration

Edit `_config.yml` to customize:
- Site title and description
- Social media links
- Navigation menu items
- Theme settings

### Styling

To customize the appearance:
1. Create `assets/css/style.scss`
2. Import minima and add custom styles:
   ```scss
   ---
   ---
   @import "minima";
   
   // Custom styles here
   ```

### Adding Pages

1. Create a new `.md` file in the root directory
2. Add front matter with `layout: page`
3. Add the page to navigation in `_config.yml`

## 📄 Pages

- **Home** (`/`) - Lists all blog posts
- **About** (`/about/`) - Personal information and background
- **Talks** (`/talks/`) - Conference presentations and speaking engagements
- **Resources** (`/resources/`) - Useful tools, projects, and links

## 🔧 Deployment

This site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

### Manual Deployment Steps

1. Make your changes
2. Test locally
3. Commit and push:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin main
   ```

GitHub Actions will automatically build and deploy your site.

## 📈 Features

- ✅ Responsive design
- ✅ SEO optimized
- ✅ Social media integration
- ✅ RSS feed
- ✅ Syntax highlighting
- ✅ Mobile-friendly
- ✅ Fast loading
- ✅ Accessible

## 🤝 Contributing

If you find any issues or have suggestions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

- **Email**: your-email@example.com
- **Twitter**: [@yourusername](https://twitter.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourusername)
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

Built with ❤️ using Jekyll and GitHub Pages