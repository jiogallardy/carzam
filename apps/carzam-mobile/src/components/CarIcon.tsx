import Svg, { Path } from "react-native-svg";

type Props = { size?: number; color?: string };

// Stylized sports-coupe side profile. Flat fill, no detail — reads at any size.
export function CarIcon({ size = 32, color = "#fff" }: Props) {
  return (
    <Svg viewBox="0 0 64 36" width={size} height={size} fill={color}>
      <Path d="M 4,26 Q 4,24 6,24 L 8,24 Q 9,21 12,20 L 18,18 Q 22,14 28,13 L 40,11 Q 50,11 56,14 L 60,20 Q 62,22 62,24 L 62,26 Q 62,28 60,28 L 56,28 Q 55,32 50,32 Q 45,32 44,28 L 20,28 Q 19,32 14,32 Q 9,32 8,28 L 6,28 Q 4,28 4,26 Z" />
    </Svg>
  );
}
