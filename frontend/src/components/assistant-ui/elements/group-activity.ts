export function groupAdjacentTools(parts: readonly { type: string }[]) {
  const groups: { groupKey: string | undefined; indices: number[] }[] = [];
  parts.forEach((part, index) => {
    if (part.type === "tool-call") {
      const last = groups[groups.length - 1];
      if (last?.groupKey === "tools") last.indices.push(index);
      else groups.push({ groupKey: "tools", indices: [index] });
    } else groups.push({ groupKey: undefined, indices: [index] });
  });
  return groups;
}
