import type React from "react"
import "./globals.css"
import NoiseOverlay from "@/components/NoiseOverlay"

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <NoiseOverlay />
        {children}
      </body>
    </html>
  )
}