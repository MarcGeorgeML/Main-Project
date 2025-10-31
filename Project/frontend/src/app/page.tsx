import GetStartedButton from "@/components/ui/button/GetStartedButton";

export default function Home() {
  return (
    <div className="flex flex-col justify-center items-center h-screen">
      <p className="font-semibold text-2xl">SentiCore</p>
      <GetStartedButton />
    </div>
  )
}