# -----------------------------------------------------------------------------
# Synthline Web App Dockerfile
# -----------------------------------------------------------------------------
# Base stage for shared settings
FROM node:20-alpine AS base

# -----------------------------------------------------------------------------
# Dependencies stage - only handles node modules
FROM base AS deps
WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies with clean output and no audit
RUN npm ci --no-audit --no-fund && \
    npm cache clean --force

# -----------------------------------------------------------------------------
# Builder stage - handles application build
FROM base AS builder
WORKDIR /app

# Set Next.js to build in standalone mode
ENV NEXT_OUTPUT_STANDALONE=true

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy application source
COPY . .

# Build the application
RUN npm run build

# -----------------------------------------------------------------------------
# Final stage
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

# Create a non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs && \
    chown -R nextjs:nodejs /app

# Copy only the necessary files from the builder stage
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Set proper permissions
RUN chmod -R 550 /app && \
    chmod -R 770 /app/.next

# Switch to non-root user
USER nextjs

# Expose the application port
EXPOSE 3000

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# Command to run the application
CMD ["node", "server.js"] 