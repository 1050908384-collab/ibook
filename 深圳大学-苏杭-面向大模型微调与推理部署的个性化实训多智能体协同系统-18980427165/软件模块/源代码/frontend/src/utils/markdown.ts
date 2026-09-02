/**
 * Repairs a legacy shorthand used by some stored learning materials: a single
 * backslash followed by a space inside an aligned math environment was meant
 * to be a LaTeX row break. KaTeX requires two backslashes for that operation.
 */
export function normalizeMarkdownMath(content: string): string {
  return content.replace(/\$\$([\s\S]*?)\$\$/g, (_match, formula: string) => {
    const normalized = formula.replace(
      /(\\begin\{(?:aligned|align\*?|gathered|array|cases)\}[\s\S]*?\\end\{(?:aligned|align\*?|gathered|array|cases)\})/g,
      (environment: string) => environment.replace(/(^|[^\\])\\ (?=[&a-zA-Z])/g, '$1\\\\ ')
    );
    return `$$${normalized}$$`;
  });
}
